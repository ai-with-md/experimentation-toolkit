# Decision memo: Hillstrom email test

**Decision: send the men's email to everyone on the list** of customers like the ones in this test (people who bought in the past year).

## Why

- **It makes money.** Expected profit is **$250 per 1,000 emails** compared with sending nothing (plausible range $153 to $351).
- **It is the better of the two emails.** It has a **99% chance** of being the best option.
- **Being wrong would cost almost nothing.** The expected loss from choosing it is **$0.20 per 1,000 emails**.
- **It passes the rule set before the results were seen:** at least 80% chance of being best, and at most $10 expected loss per 1,000 emails.

## Options compared

| Option | Expected profit per 1,000 emails | Chance best | Expected loss per 1,000 emails |
|---|---|---|---|
| Men's email | $250 | 99% | $0.20 |
| Women's email | $128 | 1% | $121 |
| No email | $0 | 0% | $250 |

## How robust it is

- The men's email beats the women's under **every** margin (10% to 60%) and email cost (0 to 30 cents) tested.
- At the assumed 35% margin, emailing stays worth it up to about **22.5 cents per email**. The assumed cost is $0.02.

## What this does not cover

- **Unsubscribes were not measured.** A rise in unsubscribes could outweigh the revenue. Check them in the first send.
- **Two weeks only.** Longer-term effects on repeat buying are unknown.
- **Margin and email cost are assumptions.** The decision holds across a wide range, but should be confirmed with finance.

## Who should get it

**Everyone on the list.** Two uplift models could not tell strong responders from weak ones on customers they had not seen (notebook 08), and emailing everyone made more profit than every targeting policy tested (notebook 09). The email pays for itself in every customer segment we can measure.

## Next

- Measure unsubscribes in the first send and add them to the email's cost.
- In the next test, log each customer's spend in the weeks before the send, so variance reduction can work.
- Consider purchases as the primary metric: it needs about half as many customers as revenue to catch the same lift (notebook 04).
- Revisit targeting only if the email gets much more expensive, or a test has many more buyers (notebook 09).
