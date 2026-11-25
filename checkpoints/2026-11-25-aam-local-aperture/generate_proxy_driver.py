"""Write the matched proxy-risk study driver from the fixed acquisition comparison."""
from pathlib import Path
D = Path(__file__).resolve().parent
code = (D / "minimal.py").read_text(encoding="utf-8")
code = code.replace("MINIMAL_", "PROXY_").replace("new-minimal-", "new-proxy-")
code = code.replace('"minimal.py", "PROXY_PROTOCOL_ZH.md", "minimal_geometry.py", "minimal_kernels.py",',
                    '"proxy.py", "PROXY_PROTOCOL_ZH.md", "proxy_geometry.py", "proxy_kernels.py",')
code = code.replace('"generate_minimal.py", "mobile_capture.py"', '"generate_proxy_driver.py", "generate_proxy.py", "mobile_capture.py"')
code = code.replace('"REPLICATION_RESULT.json", "REPLICATION_C_INDEPENDENT_AUDIT.json"',
                    '"MINIMAL_RESULT.json", "REPLICATION_RESULT.json", "REPLICATION_C_INDEPENDENT_AUDIT.json"')
code = code.replace("import minimal_kernels", "import proxy_kernels").replace(
                    "minimal_kernels.advance_minimal", "proxy_kernels.advance_proxy")
code = code.replace('"calibration ablation"', '"independent geometry proxy"')
code = code.replace('"count_rbf"', '"proxy_rbf"').replace('"marked_counts"', '"proxy_conditional"')
j = code.index("    seen = count(m).reshape")
stop = code.index("\n\ndef freeze():", j)
code = code[:j] + '    return full.features(m, x, "events")\n' + code[stop:]
j = code.index("    directions = count(raw[2])")
stop = code.index("    for key, mode", j)
code = code[:j] + '''    lr = full.conditional(check[2], True)
    scale = min((.25, .65, 1., 1.5, 2.), key=lambda s: np.mean((expit(lr*s)-dev_labels)**2))
    data, char = {}, {"mark_scale": scale}
''' + code[stop:]
code = code.replace('marked_prob(count(m), bank["mark_log"], which["mark_scale"])',
                    'expit(which["mark_scale"] * full.conditional(m, True))')
code = code.replace('"marks_full_then_minimal"', '"marks_full_then_proxy"')
with (D / "proxy.py").open("x", encoding="utf-8") as f:
    f.write(code)

