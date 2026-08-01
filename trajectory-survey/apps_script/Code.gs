/**
 * Receives trajectory-rating responses: appends them to this spreadsheet and emails them on.
 *
 * The survey is a static page on GitHub Pages, so it has no server of its own. This web app
 * is the one piece that runs somewhere — it takes the POST the Submit button makes, writes
 * two rows-per-response into the sheet, and mails a copy with the .json attached so the
 * originals can go straight into trajectory-survey/responses/ for score_survey.py.
 *
 * Deploy (once):
 *   1. sheets.new  →  name it, e.g. "D2I trajectory rating responses"
 *   2. Extensions → Apps Script.  Delete the stub, paste this file, Save.
 *   3. Deploy → New deployment → type "Web app"
 *        Execute as:      Me
 *        Who has access:  Anyone                 ← must be "Anyone", not "Anyone with Google account"
 *   4. Authorise when prompted (it asks for Sheets + Gmail: it writes rows and sends mail).
 *   5. Copy the /exec URL and paste it into ENDPOINT in trajectory-survey/index.html.
 *   6. Open the /exec URL in a browser — it should answer "ready".
 *
 * Re-deploy after editing: Deploy → Manage deployments → edit → Version "New version".
 * Editing the code alone changes nothing at the old URL until you do that.
 */

/** Where to mail each response. Set to '' to write to the sheet only. */
const NOTIFY = 'shaolinx@usc.edu';

const SHEET_RESPONSES = 'responses';   // one row per person
const SHEET_RATINGS = 'ratings';       // one row per (trajectory, criterion) — the analysable shape
const MAX_BYTES = 400000;              // a real response is a few KB; this only stops abuse

// A Sheets *cell* holds at most 50 000 characters and silently truncates past that, which
// would leave a raw_json that no longer parses — discovered only when the analysis is run,
// long after the respondent has gone. Nothing is lost when it trips: the full body is
// still attached to the notification email.
const CELL_LIMIT = 45000;

// A response records what the respondent did and nothing about the model — no scores, no
// agreement — so there is nothing here to lift out of it. Agreement is computed offline,
// by score_survey.py, from these answers and a scoring run.
const RESPONSE_COLS = [
  'received_at', 'id', 'name', 'email', 'background', 'started_at', 'finished_at',
  'n_rated', 'n_skipped', 'n_ratings', 'built_at', 'seed', 'rubric_version',
  'user_agent', 'raw_json', 'raw_truncated',
];

const RATING_COLS = [
  'received_at', 'id', 'name', 'item', 'content', 'dataset', 'system', 'criterion',
  'score', 'note', 'answered_at', 'ms',
];


function cell(v) {
  return v === null || v === undefined ? '' : v;
}


/** A quick health check: open the /exec URL in a browser to confirm the deployment. */
function doGet() {
  const n = tab(SHEET_RESPONSES, RESPONSE_COLS).getLastRow() - 1;
  return ContentService
    .createTextOutput('ready — ' + Math.max(0, n) + ' response(s) received so far')
    .setMimeType(ContentService.MimeType.TEXT);
}


function doPost(e) {
  // Two respondents finishing at the same moment would otherwise interleave their appends.
  const lock = LockService.getScriptLock();
  try {
    lock.waitLock(30000);
  } catch (err) {
    return reply({ok: false, error: 'busy, try again'});
  }

  try {
    if (!e || !e.postData || !e.postData.contents) {
      return reply({ok: false, error: 'empty body'});
    }
    const body = e.postData.contents;
    if (body.length > MAX_BYTES) {
      return reply({ok: false, error: 'too large'});
    }

    var R;
    try {
      R = JSON.parse(body);
    } catch (err) {
      return reply({ok: false, error: 'bad json'});
    }
    if (!R || !Array.isArray(R.answers) || !R.answers.length) {
      return reply({ok: false, error: 'no ratings in that response'});
    }

    const responses = tab(SHEET_RESPONSES, RESPONSE_COLS);
    const id = String(R.id || '');

    // The page retries opaquely when the browser blocks it from reading our reply, so the
    // same response can legitimately arrive twice. The submission id makes that harmless.
    if (id && alreadyHave(responses, id)) {
      return reply({ok: true, duplicate: true});
    }

    const now = new Date();
    const who = (R.respondent || {});
    const name = String(who.name || 'anonymous');
    const survey = (R.survey || {});
    const geval = survey.geval || {};
    const skipped = R.answers.filter(function (a) { return a.skipped; });

    var nRatings = 0;
    R.answers.forEach(function (a) {
      nRatings += Object.keys(a.scores || {}).length;
    });

    const truncated = body.length > CELL_LIMIT;
    responses.appendRow([
      now, id, name, String(who.email || ''), String(who.background || ''),
      String(R.started_at || ''), String(R.finished_at || ''),
      R.answers.length - skipped.length, skipped.length, nRatings,
      String(survey.built_at || ''), survey.seed === undefined ? '' : survey.seed,
      String(geval.rubric_version || ''),
      String(R.user_agent || ''),
      truncated ? body.slice(0, CELL_LIMIT) : body,
      truncated ? 'yes — use the emailed attachment' : '',
    ]);

    const ratings = tab(SHEET_RATINGS, RATING_COLS);
    const rows = [];
    R.answers.forEach(function (a) {
      const scores = a.scores || {};
      const keys = Object.keys(scores);
      const head = [now, id, name, String(a.item || ''), String(a.content || ''),
                    String(a.dataset || ''), String(a.system || '')];
      if (!keys.length) {
        rows.push(head.concat(['(skipped)', '', String(a.note || ''),
                               String(a.answered_at || ''), cell(a.ms)]));
        return;
      }
      keys.forEach(function (k) {
        rows.push(head.concat([k, scores[k], String(a.note || ''),
                               String(a.answered_at || ''), cell(a.ms)]));
      });
    });
    if (rows.length) {
      ratings.getRange(ratings.getLastRow() + 1, 1, rows.length, RATING_COLS.length)
        .setValues(rows);
    }

    if (NOTIFY) {
      notify(R, name, nRatings, skipped.length, body);
    }
    return reply({ok: true, received: nRatings});

  } catch (err) {
    // Logged to the Apps Script execution log, and reported back so the page can say why.
    console.error(err);
    return reply({ok: false, error: String(err)});
  } finally {
    lock.releaseLock();
  }
}


/** Mail one response on, with the .json attached — the file score_survey.py reads. */
function notify(R, name, nRatings, nSkipped, body) {
  const survey = (R.survey || {});
  const file = 'd2i-rating_'
    + name.replace(/[^A-Za-z0-9._-]+/g, '-') + '_'
    + String(R.started_at || '').slice(0, 10) + '.json';
  const lines = [
    'name       : ' + name,
    'email      : ' + String((R.respondent || {}).email || '—'),
    'background : ' + String((R.respondent || {}).background || '—'),
    '',
    'rated      : ' + (R.answers.length - nSkipped) + ' trajectories, ' + nRatings + ' ratings',
    'skipped    : ' + nSkipped,
    '',
    'started    : ' + String(R.started_at || '—'),
    'finished   : ' + String(R.finished_at || '—'),
    'build      : ' + String(survey.built_at || '—') + '  seed ' +
      (survey.seed === undefined ? '—' : survey.seed),
    '',
    'The attachment is the response file. Save it into trajectory-survey/responses/ and run:',
    '  python3 trajectory-survey/score_survey.py trajectory-survey/responses/ --per-respondent',
  ];
  MailApp.sendEmail({
    to: NOTIFY,
    subject: 'D2I trajectory rating — ' + name + ' (' + nRatings + ' ratings)',
    body: lines.join('\n'),
    attachments: [Utilities.newBlob(body, 'application/json', file)],
  });
}


/** The named sheet, created with its header row if this is the first response. */
function tab(title, cols) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = ss.getSheetByName(title);
  if (!sh) {
    sh = ss.insertSheet(title);
  }
  if (sh.getLastRow() === 0) {
    sh.appendRow(cols);
    sh.getRange(1, 1, 1, cols.length).setFontWeight('bold');
    sh.setFrozenRows(1);
  }
  return sh;
}


/** Has this submission id been stored already? Column B of the responses sheet. */
function alreadyHave(sheet, id) {
  const last = sheet.getLastRow();
  if (last < 2) {
    return false;
  }
  const seen = sheet.getRange(2, 2, last - 1, 1).getValues();
  for (var i = 0; i < seen.length; i++) {
    if (String(seen[i][0]) === id) {
      return true;
    }
  }
  return false;
}


function reply(payload) {
  return ContentService
    .createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}
