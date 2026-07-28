/**
 * Receives survey responses: appends them to this spreadsheet and emails them on.
 *
 * The survey is a static page on GitHub Pages, so it has no server of its own. This web app
 * is the one piece that runs somewhere — it takes the POST the Submit button makes, writes
 * two rows-per-response into the sheet, and mails a copy with the .json attached so the
 * originals can go straight into survey/responses/ for score_survey.py.
 *
 * Deploy (once):
 *   1. sheets.new  →  name it, e.g. "D2I survey responses"
 *   2. Extensions → Apps Script.  Delete the stub, paste this file, Save.
 *   3. Deploy → New deployment → type "Web app"
 *        Execute as:      Me
 *        Who has access:  Anyone                 ← must be "Anyone", not "Anyone with Google account"
 *   4. Authorise when prompted (it asks for Sheets + Gmail: it writes rows and sends mail).
 *   5. Copy the /exec URL and paste it into ENDPOINT in survey/index.html.
 *   6. Open the /exec URL in a browser — it should answer "ready".
 *
 * Re-deploy after editing: Deploy → Manage deployments → edit → Version "New version".
 * Editing the code alone changes nothing at the old URL until you do that.
 */

/** Where to mail each response. Set to '' to write to the sheet only. */
const NOTIFY = 'shaolinx@usc.edu';

const SHEET_RESPONSES = 'responses';   // one row per person
const SHEET_ANSWERS = 'answers';       // one row per answer — the analysable shape
const MAX_BYTES = 400000;              // a real response is a few KB; this only stops abuse

const RESPONSE_COLS = [
  'received_at', 'id', 'name', 'email', 'background', 'started_at', 'finished_at',
  'n_answered', 'n_picks', 'n_terminate', 'n_skipped', 'seed', 'built_at',
  'user_agent', 'raw_json',
];
const ANSWER_COLS = [
  'received_at', 'id', 'name', 'decision', 'terminate', 'slot', 'choice', 'note', 'answered_at',
];


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
    if (!R || !Array.isArray(R.verdicts) || !R.verdicts.length) {
      return reply({ok: false, error: 'no answers in that response'});
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
    const picks = R.verdicts.filter(function (v) { return v.slot !== null && v.slot !== undefined; });
    const terms = R.verdicts.filter(function (v) { return v.terminate === true; });
    const skips = R.verdicts.filter(function (v) { return v.choice === 's'; });

    responses.appendRow([
      now, id, name, String(who.email || ''), String(who.background || ''),
      String(R.started_at || ''), String(R.finished_at || ''),
      R.verdicts.length, picks.length, terms.length, skips.length,
      survey.seed === undefined ? '' : survey.seed, String(survey.built_at || ''),
      String(R.user_agent || ''), body,
    ]);

    const answers = tab(SHEET_ANSWERS, ANSWER_COLS);
    const rows = R.verdicts.map(function (v) {
      return [
        now, id, name, String(v.decision || ''),
        v.terminate === null || v.terminate === undefined ? '' : v.terminate,
        v.slot === null || v.slot === undefined ? '' : v.slot,
        String(v.choice || ''), String(v.note || ''), String(v.ts || ''),
      ];
    });
    if (rows.length) {
      answers.getRange(answers.getLastRow() + 1, 1, rows.length, ANSWER_COLS.length).setValues(rows);
    }

    if (NOTIFY) {
      notify(R, name, picks.length, terms.length, skips.length, body);
    }
    return reply({ok: true, received: R.verdicts.length});

  } catch (err) {
    // Logged to the Apps Script execution log, and reported back so the page can say why.
    console.error(err);
    return reply({ok: false, error: String(err)});
  } finally {
    lock.releaseLock();
  }
}


/** Mail one response on, with the .json attached — the file score_survey.py reads. */
function notify(R, name, nPicks, nTerm, nSkip, body) {
  const survey = (R.survey || {});
  const file = 'd2i-survey_'
    + name.replace(/[^A-Za-z0-9._-]+/g, '-') + '_'
    + String(R.started_at || '').slice(0, 10) + '.json';
  const lines = [
    'name       : ' + name,
    'email      : ' + String((R.respondent || {}).email || '—'),
    'background : ' + String((R.respondent || {}).background || '—'),
    '',
    'answers    : ' + R.verdicts.length,
    '  picks    : ' + nPicks,
    '  terminate: ' + nTerm,
    '  skipped  : ' + nSkip,
    '',
    'started    : ' + String(R.started_at || '—'),
    'finished   : ' + String(R.finished_at || '—'),
    'build seed : ' + (survey.seed === undefined ? '—' : survey.seed),
    '',
    'The attachment is the response file. Save it into survey/responses/ and run:',
    '  python3 survey/score_survey.py survey/responses/ --per-respondent',
    '',
    'Slots are blinded: the shuffle that resolves them lives only in',
    'survey/private/decisions.json, so the scorer is the only thing that can read them.',
  ];
  MailApp.sendEmail({
    to: NOTIFY,
    subject: 'D2I survey response — ' + name + ' (' + R.verdicts.length + ' answers)',
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
