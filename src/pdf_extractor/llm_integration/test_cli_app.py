from typer.testing import CliRunner
from pdf_extractor.llm_integration._archive.cli_app import app  

runner = CliRunner()

def test_basic_functionality():
    result = runner.invoke(app, ["your-command", "--option", "value"])
    assert result.exit_code == 0
    assert "Expected output" in result.stdout