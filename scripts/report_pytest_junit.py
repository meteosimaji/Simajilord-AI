"""Run the pytest JUnit collection gate used by GitHub Actions."""

from simajilord.diagnostics.ci_junit import main

if __name__ == "__main__":
    raise SystemExit(main())
