import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def test_sections_and_legacy_report(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'training_app.py'), default_timeout=90).run()
        self.assertFalse(list(app.exception), [e.message for e in app.exception])
        replay=[x for x in app.number_input if x.label=='Improvement test pair row']
        if replay:
            replay[0].set_value(1).run()
            self.assertFalse(list(app.exception), [e.message for e in app.exception])
        chooser = next(x for x in app.multiselect if x.label == 'Choose completed runs')
        legacy = next(x for x in chooser.options if 's3_seed_00_cpu_run' in x)
        chooser.set_value([legacy]).run()
        self.assertFalse(list(app.exception), [e.message for e in app.exception])
        for section in ['Preprocessing', 'Classical vs Quantum', 'Project roadmap']:
            app.sidebar.radio[0].set_value(section).run()
            self.assertFalse(list(app.exception), [e.message for e in app.exception])
            if section == 'Classical vs Quantum':
                chooser = app.selectbox(key='kernel_report')
                for index in range(len(chooser.options)):
                    app.selectbox(key='kernel_report').select_index(index).run()
                    self.assertFalse(list(app.exception), [e.message for e in app.exception])
                if app.number_input:
                    app.number_input[0].set_value(1).run()
                    self.assertFalse(list(app.exception), [e.message for e in app.exception])


if __name__ == '__main__':
    unittest.main()
