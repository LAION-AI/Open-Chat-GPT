## Adding new locales to i18n

This guide will help you add a new locale to the `i18n` setup.

### Prerequisites

- An up-to-date branch with the `main` branch.
- Familiarity with `i18n`, `react-i18next`, and `next-i18next` libraries is beneficial.

### Adding a new language

1. Determine the language and country codes using `ISO 639-1`. For example, `en` for English.
1. Create a new directory within the `public/locales` directory using the language and country codes as the name, for
   example `en`.
1. Copy all the files from the `en` directory into the newly created directory.
1. Edit the copied the text in the copied files with the desired language.
1. Add the new language to the list in `next-i18next.config.js` if it does not already exist.
1. Follow the instructions in [Website README](<[README.md](../../../website/README.md)>) to run and test the new
   language by changing the active locale in the application and verifying that all translated keys are properly
   displayed.
1. Commit your changes and open a pull request against the `main` branch for review.

### Editing existing translation files

When editing existing translations, follow these rules:

1. English translations are required, and other locales fall back to them.
1. Keep translation keys in alphabetical order.
1. Add all translations for higher-level components (e.g. `Layout.ts`) in `common.json` to prevent hydration issues.
1. Add reused translation keys in `common.json`.
1. Split translation files into separate files by feature or route.

### Finding missing translations

A script can be used to find missing and potentially untranslated locale files. Run the script from the root dir using
`python scripts/frontend-development/find-missing-locales.py`.

You may pass any languages you want to include in the search to the script like so:\
`python scripts/frontend-development/find-missing-locales.py de ja`\
This would only show missing/untranslated strings in the German (de) and Japanese (ja) locale files.\
Passing nothing will default to searching all locales instead.

If you have any questions or need further assistance, please reach out.

### Adding missing translations

A script can be used to add missing entries to locale files. Run the script from the root dir using
`python scripts/frontend-development/add_missing_json_locale_entries.py -h` for help on how to use it.
