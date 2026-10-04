"""LLM prompts used across services.

Kept in one place so tests can import and compare exact strings, and so that
changes to prompt wording are tracked in a single file.
"""
from __future__ import annotations


INTENT_ROUTER_PROMPT = """You are an intent classifier for a personal Telegram assistant.
Classify the user's message into exactly one of:
- "reminder" — user wants a time-bound notification (one-off OR recurring),
    e.g. "напомни завтра в 14 позвонить",
         "напомни через 10 минут проверить плиту",
         "напоминай каждую пятницу в 18:00 закрывать отчёт",
         "remind me at 18:00 to take the medicine".
- "task" — user wants to add an open-ended TODO without a fixed time,
    typically with category and/or priority,
    e.g. "добавь задачу купить молоко",
         "задача категории work: подготовить презентацию до пятницы",
         "срочно подготовить отчёт",
         "add a task to buy a present for Masha".
- "subscription" — user wants to record a recurring service charge
    (Spotify/Netflix/ChatGPT/insurance/mobile etc.),
    typically with amount, currency and a payment day,
    e.g. "Spotify 169₽ с Сбера каждое 15 число",
         "Netflix 499 руб ежемесячно",
         "ChatGPT Plus 20 долларов каждый месяц",
         "subscription: iCloud 99 руб 20-го".
- "unknown" — anything else (greetings, small talk, questions, file uploads, etc.).

Disambiguation rules (apply in order):
1. If the message contains «напомн…» / "remind me" / explicit time of day OR
   a recurring schedule («каждый…», «every…») WITHOUT a money amount → reminder.
2. If the message has a money amount (₽/руб/$/€/usd/eur) AND a periodic day
   («каждое N число», «ежемесячно», «monthly», «каждый месяц») → subscription.
3. If the message has a money amount but no periodicity → task (e.g. "купить
   подарок Маше за 3000₽" — это задача с деньгами, не подписка).
4. If the message starts with «задача …» / «добавь задачу» / «task: …» / has
   a clear deadline («до пятницы») without «напомн…» → task.
5. Otherwise → unknown.

Return STRICTLY a single JSON object with keys:
- "intent": one of "reminder" | "task" | "subscription" | "unknown"
- "confidence": float in [0.0, 1.0]

No additional text, no markdown, no code blocks, no explanations. Only the JSON object.

User message: {text}
"""


REMINDER_PARSER_PROMPT = """You are a reminder parser for a personal Telegram assistant.
Extract the title and the target datetime from the user's message.

Current datetime (Europe/Moscow): {now}
Timezone reference: Europe/Moscow (UTC+3)

Return STRICTLY a single JSON object with keys:
- "title": string — concise description of what to remind about (no leading "напомни", no trailing punctuation)
- "when_at": string — ISO 8601 datetime with timezone offset "+03:00",
  e.g. "2026-04-24T14:00:00+03:00"

Rules:
- Interpret relative times ("завтра", "через час", "сегодня вечером", "tomorrow") relative to the current datetime above.
- "when_at" MUST be strictly in the future relative to the current datetime.
- If the message lacks an extractable datetime, return {{"title": "", "when_at": ""}}.
- Return ONLY the JSON object — no markdown, no code blocks, no prose.

User message: {text}
"""


UNIFIED_REMINDER_PROMPT = """You are both an intent classifier AND a reminder parser for a personal Telegram assistant.
In a single step, decide whether the user's message is a reminder request
(one-off OR recurring) and extract its structure.

Current datetime (Europe/Moscow): {now}
Timezone reference: Europe/Moscow (UTC+3)

Return STRICTLY a single JSON object with these keys:
- "intent":  one of
    "reminder_single"    — one-off reminder, e.g. "напомни завтра в 14 позвонить".
    "reminder_recurring" — repeating reminder, e.g. "каждую пятницу в 18",
                           "каждое 1 число в 9", "каждые 2 недели во вторник".
    "unknown"            — anything else (greetings, small talk, questions, etc.).
- "confidence": float in [0.0, 1.0]
- "title":   string — concise description of what to remind about
             (empty "" if intent == "unknown"; no leading "напомни").
- "when_at": string — ISO 8601 datetime with offset "+03:00",
             e.g. "2026-04-24T14:00:00+03:00"
             (NON-empty only when intent == "reminder_single"; "" otherwise).
- "rrule":   string — RFC 5545 RRULE *without* DTSTART prefix,
             using only these components: FREQ, INTERVAL, BYDAY,
             BYMONTHDAY, BYHOUR, BYMINUTE, UNTIL, COUNT.
             (NON-empty only when intent == "reminder_recurring"; "" otherwise).
- "exdates": array of "YYYY-MM-DD" strings — explicit dates to skip
             (empty array [] if none).
- "exclude_holidays": one of "none" | "ru" | "ru+custom" —
             "ru" when the user says "кроме праздников РФ" or similar;
             "none" otherwise.

Rules:
- Interpret relative times ("завтра", "через час", "сегодня вечером", "tomorrow")
  relative to the current datetime above.
- "when_at" MUST be strictly in the future relative to the current datetime.
- Use BYDAY values: MO, TU, WE, TH, FR, SA, SU.
- Return ONLY the JSON object — no markdown, no code blocks, no prose.

Few-shot examples (for calibration — do NOT include them in output):

User: "напомни завтра в 14 позвонить в страховую"
  → {{"intent":"reminder_single","confidence":0.98,"title":"позвонить в страховую","when_at":"<tomorrow>T14:00:00+03:00","rrule":"","exdates":[],"exclude_holidays":"none"}}

User: "напомни через 10 минут проверь плиту"
  → {{"intent":"reminder_single","confidence":0.98,"title":"проверь плиту","when_at":"<now+10min>","rrule":"","exdates":[],"exclude_holidays":"none"}}

User: "напоминай каждую пятницу в 18:00 сдать отчёт"
  → {{"intent":"reminder_recurring","confidence":0.97,"title":"сдать отчёт","when_at":"","rrule":"FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0","exdates":[],"exclude_holidays":"none"}}

User: "каждое 1 число в 9 утра оплатить ипотеку"
  → {{"intent":"reminder_recurring","confidence":0.97,"title":"оплатить ипотеку","when_at":"","rrule":"FREQ=MONTHLY;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0","exdates":[],"exclude_holidays":"none"}}

User: "каждые 2 недели во вторник в 12 к стоматологу"
  → {{"intent":"reminder_recurring","confidence":0.95,"title":"к стоматологу","when_at":"","rrule":"FREQ=WEEKLY;INTERVAL=2;BYDAY=TU;BYHOUR=12;BYMINUTE=0","exdates":[],"exclude_holidays":"none"}}

User: "каждую пятницу в 18, кроме праздников РФ, еженедельный отчёт"
  → {{"intent":"reminder_recurring","confidence":0.96,"title":"еженедельный отчёт","when_at":"","rrule":"FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0","exdates":[],"exclude_holidays":"ru"}}

User: "по будням в 9 утра утренняя планёрка"
  → {{"intent":"reminder_recurring","confidence":0.96,"title":"утренняя планёрка","when_at":"","rrule":"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0","exdates":[],"exclude_holidays":"none"}}

User: "привет"
  → {{"intent":"unknown","confidence":0.05,"title":"","when_at":"","rrule":"","exdates":[],"exclude_holidays":"none"}}

User message: {text}
"""


DOC_CLASSIFIER_PROMPT = """You are a document type classifier for a personal Telegram assistant.
Classify the attached document text into exactly one of these closed types:
- "passport"  — Russian internal passport (паспорт гражданина РФ).
- "snils"     — СНИЛС (страховое свидетельство пенсионного страхования).
- "driving"   — водительское удостоверение.
- "inn"       — свидетельство ИНН (идентификационный номер налогоплательщика).
- "policy"    — страховой полис (ОМС / ДМС / ОСАГО / КАСКО / life).
- "contract"  — договор (услуг, аренды, купли-продажи, трудовой и т.п.).
- "other"     — anything else that does not clearly match the six types above.

Return STRICTLY a single JSON object with one key:
- "type": one of the 7 literal values above (no synonyms, no translations).

No additional text, no markdown, no code blocks, no explanations. Only the JSON object.

Document text (may be truncated):
{text}
"""


TASK_PARSER_PROMPT = """You are a task parser for a personal Telegram assistant.
The user's message is already known to be a TODO/task — extract its title,
optional deadline, priority and category.

Current datetime (Europe/Moscow): {now}
Timezone reference: Europe/Moscow (UTC+3)

Return STRICTLY a single JSON object with keys:
- "title":    string — concise description (no leading "задача"/"добавь задачу"/"task",
              no trailing punctuation).
- "deadline": string — ISO 8601 date OR datetime with offset "+03:00",
              e.g. "2026-05-01" or "2026-05-01T18:00:00+03:00".
              "" if no deadline mentioned.
- "priority": one of "low" | "med" | "high".
              "high" if the message contains «срочно», «asap», «urgent»,
              «важно», «не забудь», «ASAP» — case-insensitive.
              "low" if «когда-нибудь», «не срочно», «later», «backlog».
              Otherwise "med".
- "category": short snake_case string — typical categories: "general", "work",
              "home", "shopping", "health", "personal", "study", "finance".
              Use "general" if not obvious. Always lowercase.

Rules:
- If a deadline is given, MUST be strictly in the future relative to current datetime.
- For relative dates («к пятнице», «до конца недели», «завтра», «through Monday»),
  pick the next matching date strictly after current datetime.
- "deadline" empty string "" is allowed (open-ended task).
- Return ONLY the JSON object — no markdown, no code blocks, no prose.

Few-shot examples (do NOT include in output):

User: "добавь задачу купить молоко"
  → {{"title":"купить молоко","deadline":"","priority":"med","category":"shopping"}}

User: "купить молоко к пятнице, категория home"
  → {{"title":"купить молоко","deadline":"<next-friday>","priority":"med","category":"home"}}

User: "срочно подготовить отчёт до пятницы, work"
  → {{"title":"подготовить отчёт","deadline":"<next-friday>","priority":"high","category":"work"}}

User: "задача: записаться к стоматологу"
  → {{"title":"записаться к стоматологу","deadline":"","priority":"med","category":"health"}}

User: "не срочно — почитать книгу про React"
  → {{"title":"почитать книгу про React","deadline":"","priority":"low","category":"study"}}

User message: {text}
"""


SUBSCRIPTION_PARSER_PROMPT = """You are a subscription parser for a personal Telegram assistant.
The user's message is already known to be a recurring paid subscription —
extract service name, amount, currency, optional card and payment day.

Return STRICTLY a single JSON object with keys:
- "service":             string — concise service name (e.g. "Spotify",
                         "Netflix", "ChatGPT Plus", "iCloud", "Yota mobile").
                         Capitalised as the user wrote it.
- "amount":              number — numeric amount, e.g. 169, 499, 19.99.
                         No currency symbol. Use float for decimals.
- "currency":            one of "RUB" | "USD" | "EUR" | "GBP".
                         Default RUB if unclear.
                         Map "₽"/"руб"/"рублей"/"р." → "RUB".
                         Map "$"/"usd"/"долларов"/"бакса" → "USD".
                         Map "€"/"eur"/"евро" → "EUR".
                         Map "£"/"gbp"/"фунтов" → "GBP".
- "card":                string — payment card/bank as the user mentioned
                         it ("Сбер", "Тинькофф", "Visa", "Apple Pay").
                         "" if not mentioned.
- "recurrence":          one of "monthly" | "yearly" | "weekly".
                         Default "monthly".
                         "yearly" for «годовая», «в год», «annually».
                         "weekly" for «еженедельно», «каждую неделю», «weekly».
- "day_of_month":        integer 1..31 — payment day. Required when recurrence="monthly"
                         (default 1 if user said only "каждый месяц" without day).
                         Use null when recurrence != "monthly".
- "notify_before_days":  integer ≥ 0 — days before charge to remind. Default 2.
                         If user says «без уведомления» / «не напоминай» → 0.
                         If user says «напомни за неделю» → 7.

Rules:
- Service name MUST be present and non-empty.
- Amount MUST be > 0.
- Return ONLY the JSON object — no markdown, no code blocks, no prose.

Few-shot examples (do NOT include in output):

User: "Netflix 499 руб Тинькофф каждое 20 число"
  → {{"service":"Netflix","amount":499,"currency":"RUB","card":"Тинькофф","recurrence":"monthly","day_of_month":20,"notify_before_days":2}}

User: "Spotify 169₽ с Сбера каждое 15 число"
  → {{"service":"Spotify","amount":169,"currency":"RUB","card":"Сбер","recurrence":"monthly","day_of_month":15,"notify_before_days":2}}

User: "ChatGPT Plus 20 долларов ежемесячно"
  → {{"service":"ChatGPT Plus","amount":20,"currency":"USD","card":"","recurrence":"monthly","day_of_month":1,"notify_before_days":2}}

User: "iCloud 99 руб 20-го, напомни за неделю"
  → {{"service":"iCloud","amount":99,"currency":"RUB","card":"","recurrence":"monthly","day_of_month":20,"notify_before_days":7}}

User: "хостинг Hetzner 12 евро в год, без напоминания"
  → {{"service":"Hetzner","amount":12,"currency":"EUR","card":"","recurrence":"yearly","day_of_month":null,"notify_before_days":0}}

User message: {text}
"""


RECURRING_PARSER_PROMPT = """You are a recurring-reminder parser for a personal Telegram assistant.
The user's message is already known to be a *recurring* reminder — extract
its title, RRULE, exdates and exclude_holidays flag.

Current datetime (Europe/Moscow): {now}
Timezone reference: Europe/Moscow (UTC+3)

Return STRICTLY a single JSON object with keys:
- "title":            string — concise description (no leading "напомни").
- "rrule":            string — RFC 5545 RRULE without DTSTART prefix,
                      components allowed: FREQ, INTERVAL, BYDAY, BYMONTHDAY,
                      BYHOUR, BYMINUTE, UNTIL, COUNT.
- "exdates":          array of "YYYY-MM-DD" strings (empty [] if none).
- "exclude_holidays": one of "none" | "ru" | "ru+custom".

Rules:
- BYDAY values: MO, TU, WE, TH, FR, SA, SU.
- Return ONLY the JSON — no markdown, no code blocks, no prose.

Examples:
"каждую пятницу в 18"         → {{"title":"...","rrule":"FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0","exdates":[],"exclude_holidays":"none"}}
"каждое 1 число в 9"          → {{"title":"...","rrule":"FREQ=MONTHLY;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0","exdates":[],"exclude_holidays":"none"}}
"каждые 2 недели во вторник"  → {{"title":"...","rrule":"FREQ=WEEKLY;INTERVAL=2;BYDAY=TU","exdates":[],"exclude_holidays":"none"}}
"по будням в 9"               → {{"title":"...","rrule":"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0","exdates":[],"exclude_holidays":"none"}}
"каждую пятницу, кроме праздников РФ" → {{"title":"...","rrule":"FREQ=WEEKLY;BYDAY=FR","exdates":[],"exclude_holidays":"ru"}}

User message: {text}
"""
