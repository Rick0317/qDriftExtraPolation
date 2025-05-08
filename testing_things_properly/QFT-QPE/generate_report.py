import os
from pathlib import Path

def generate_report(output_dir=".", log_file="qpe_histograms_log.txt", html_file="qpe_output_report.html"):
    """Generate an HTML report of all QPE histograms and circuit diagram."""
    
    output_path = Path(output_dir)
    log_path = output_path / log_file
    html_path = output_path / html_file

    if not log_path.exists():
        print(f"No log file found at {log_path}. No report generated.")
        return

    # Read log file to collect metadata
    with open(log_path, "r") as log:
        lines = log.readlines()

    # Prepare HTML content
    body = """<html>
    <head>
        <title>QPE Report</title>
        <style>
            .container { display: flex; flex-wrap: wrap; gap: 20px; }
            .item { flex: 1; min-width: 300px; }
            .item img { max-width: 100%; }
        </style>
    </head>
    <body>
    <h1>Quantum Phase Estimation - Report</h1><hr>
    """

    # LaTeX Explanation
    body += r"""
    <h2>Sanity Check Analysis</h2>
    <p>The controlled unitary \( U \) is defined as:</p>
    <p>
    $$ U = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix} $$
    </p>
    <p>
    For eigenstate \( |1\rangle \), we have \( U |1\rangle = e^{i\theta} |1\rangle \).
    In the ideal case, the phase \( \theta \) is given by:
    </p>
    <p>
    $$ \theta = \frac{2 \pi k}{2^n} $$
    </p>
    <p>where \( k \) is an integer and \( n \) is the number of ancilla qubits.</p>
    <p>
    <strong>Experiments:</strong><br>
    Let \( n = 3 \), so the possible values of \( k \) are \( 0, 1, 2, 3, 4, 5, 6, 7 \).
    <ul>
        <li> \( k = 1 \implies \theta = \frac{\pi}{4} \) </li>
        <li> \( k = 2 \implies \theta = \frac{\pi}{2} \) </li>
        <li> \( k = 3 \implies \theta = \frac{3\pi}{4} \) </li>
        <li> \( k = 4 \implies \theta = \pi \) </li>
        <li> \( k = 5 \implies \theta = \frac{5\pi}{4} \) </li>
        <li> \( k = 6 \implies \theta = \frac{3\pi}{2} \) </li>
        <li> \( k = 7 \implies \theta = \frac{7\pi}{4} \) </li>
    </ul>
    </p>
    <hr>
    """

    # Loop through log entries
    for line in lines:
        filename, phase, ancilla, bitstring, est_phase = line.strip().split(",")
        img_path = output_path / filename
        circuit_path = output_path / f"{filename.replace('output_', 'circuit_')}"

        if img_path.exists() and circuit_path.exists():
            body += f"""
            <div class="container">
                <div class="item">
                    <h2>Phase: {round(float(phase), 3)} | Ancilla: {ancilla}</h2>
                    <p>Measured: {bitstring} | Estimated Phase: {round(float(est_phase), 3)}</p>
                </div>
                <div class="item">
                    <img src="{circuit_path}" alt="Circuit Diagram">
                </div>
                <div class="item">
                    <img src="{filename}" alt="Histogram">
                </div>
            </div>
            <hr>
            """
        else:
            print(f"Missing image or circuit diagram for {filename}. Skipping.")
    body += "</body></html>"

    # Write the report
    with open(html_path, "w") as html_file:
        html_file.write(body)

    print(f"Report generated: {html_path.resolve()}")

if __name__ == "__main__":
    generate_report()
