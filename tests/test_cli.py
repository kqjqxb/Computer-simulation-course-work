from src.model.run import main


def test_cli_prints_table_and_trace(capsys):
    main(["--T", "200", "--runs", "3", "--trace", "--trace-limit", "10"])
    out = capsys.readouterr().out
    assert "Ймовірність пропуску секції, –" in out
    assert "±95% ДІ" in out
    assert "Протокол подій" in out
    assert "вхід" in out


def test_cli_deterministic_intervals(capsys):
    main(["--T", "60", "--int1", "5", "0", "--int2", "20", "0"])
    out = capsys.readouterr().out
    assert "Заповнено секцій, од." in out
    line = next(l for l in out.splitlines() if l.startswith("Пропущено секцій"))
    assert line.split()[-1] == "1.0"


def test_cli_buffer_rows(capsys):
    main(["--T", "500", "--runs", "2", "--buffer", "20", "60", "--trace", "--trace-limit", "5"])
    out = capsys.readouterr().out
    assert "буфери L1 = 20, L2 = 60" in out
    assert "Частка часу блокування постачальника типу 1, –" in out
    assert "Free2" in out
