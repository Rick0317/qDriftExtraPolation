import os
from pathlib import Path
from sympy import symbols, Matrix, simplify, latex
from sympy.parsing.sympy_parser import parse_expr
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def html_to_pdf(html_path: Path, pdf_path: Path):
    # Launch headless Chromium
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # file:// URL so local images/styles load
        await page.goto(html_path.resolve().as_uri())
        # wait for MathJax to finish rendering (if you’re using it)
        await page.wait_for_timeout(2000)
        # print to PDF
        await page.pdf(
        path=str(pdf_path),
        print_background=True,
        width="1900px",
        height="2080px",
        margin={"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"}
    )
        await browser.close()


def safe_float(value):
    """Convert a value to float, extracting the real part if it is complex."""
    try:
        return float(value)
    except ValueError:
        # Handle complex numbers
        complex_val = complex(value)
        return complex_val.real

def generate_report(
    output_dir='.',
    log_file='qpe_histograms_log.txt',
    html_file='qpe_output_report.html',
    pdf_file='qpe_output_report.pdf',
    only_hamiltonian_simulation=False
):
    """Generate a comprehensive QPE report including sanity checks, approximate and exact Hermitian eigenvalue estimation."""
    
    output_path = Path(output_dir)
    log_path = output_path / log_file
    html_path = output_path / html_file
    pdf_path = output_path / pdf_file

    # Expressions to embed in text
    expression_for_expected_phase = r"\(\frac{\lambda \cdot t}{2 \cdot \pi} \mod 1\)"
    expression_for_calculated_phase_from_bitstring = r"\(\frac{k}{2^n}\)"
    expression_for_calculated_energy_from_phase = r"\(\frac{2 \cdot \pi \cdot \varphi}{t}\)"

    if not log_path.exists():
        print(f"No log file found at {log_path}. No report generated.")
        return

    with open(log_path, "r") as log:
        lines = log.readlines()

    # Initialize HTML content with enhanced styles
    body = f"""<html>
    <head>
        <title>Comprehensive QPE Report</title>
        <script type=\"text/javascript\" src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ padding: 20px; border-bottom: 1px solid #ccc; }}
            .section:nth-of-type(even) {{ background-color: #f5f5f5; }}
            .container {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }}
            .item {{ flex: 1; min-width: 250px; }}
            .item img {{ max-width: 100%; display: block; margin: 0 auto; }}
            .header {{ font-weight: bold; font-size: 1.1em; }}
            .label {{ text-align: center; font-style: italic; margin-top: 5px; }}
            .matrix-container {{ display: flex; flex-direction: column; gap: 15px; }}
            .matrix-container h3 {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
    <h1>Quantum Phase Estimation - Report</h1>
    """

    # Sanity Check Section
    if not only_hamiltonian_simulation:
        body += r"""
        <div class=\"section\">
        <h2>Sanity Check Analysis</h2>
        <p>The unitary matrix is given by:</p>

        <p>$$ U = \begin{pmatrix}
                            1 & 0 \\
                            0 & e^{i\theta}
                        \end{pmatrix} $$</p>

        <p>So, for \(|\lambda \rangle = |1\rangle\) we have:</p>

        <p>$$ U |1\rangle = e^{i\theta} |1\rangle $$</p>

        <p>In nice cases, we have:</p>

        <p>$$ \theta = \frac{2 \pi k}{2^n} $$</p>

        <p>where \(k\) is an integer and \(n\) is the number of ancilla qubits. Then, value read in the ancilla register will be exactly \(k\).</p>
        </div>
        """

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 6:
                filename, phase, ancilla, bitstring, est_phase, id = parts
                circuit_filename = f"circuit_{id}.png"
                img_path = output_path / filename
                circuit_path = output_path / circuit_filename
                if img_path.exists() and circuit_path.exists():
                    body += f"""
                    <div class=\"section\">
                    <div class=\"container\">
                        <div class=\"item header\">Phase: {round(safe_float(phase), 3)} | Ancilla: {ancilla}</div>
                        <div class=\"item\">
                        <p>Measured: {bitstring}</p>
                        <p>Estimated Phase: {round(safe_float(est_phase), 3)}</p>
                        </div>
                        <div class=\"item\">
                        <img src=\"{circuit_path}\" alt=\"Circuit Diagram\">
                        <div class=\"label\">Quantum Circuit Diagram</div>
                        </div>
                        <div class=\"item\">
                        <img src=\"{filename}\" alt=\"Histogram\">
                        <div class=\"label\">Results for Sanity Check</div>
                        </div>
                    </div>
                    </div>
                    """

    # Exact Eigenvalue Estimation Section
    body += r"""
    <div class=\"section\">
      <h2>Exact Eigenvalue Estimation of Hermitian Matrices</h2>

      <p>Let \( H \) be a Hermitian matrix. Let \( |\lambda\rangle \) be an eigenvector of \( H \) with eigenvalue \( \lambda \) \( \implies e^{iHt}|\lambda\rangle = e^{i\lambda t}|\lambda\rangle \)</p>

      <p>Of course, the measurements of QPE will <em>not</em> give us a direct approximation of \( \lambda \), but rather of some "phase" \( \theta \) such that:</p>

      <p>$$ e^{i\lambda t} = e^{i\theta} \implies \lambda = \frac{\theta}{t}$$</p>


      <p>Recall that if the phase \( \theta \) happens to be of the form \(\frac{2\pi k}{2^n} \) then we can measure the ancilla qubits and get \( k \) exactly.</p>

      <p>We can "force" this to happen. Let us say we pre-compute all eigenvalues \( \lambda_1, \ldots, \lambda_n \) of \( H \) and want to estimate the eigenvalue \( \lambda_i \). Well, we can simply choose \( t \) such that:</p>

      <p>$$ \lambda_i = \frac{2\pi k}{t 2^n} $$</p>

      <p>This is equivalent to choosing:</p>

      <p>$$ t = \frac{2\pi k}{\lambda_i 2^n} $$</p>

      <hr>
    """
    ham_block = ""
    exact_block = ""
    added_ham = False
    for line in lines:
        parts = line.strip().split(",")
        if parts[0] == "exact phase":
            _, filename, t, ancilla, bitstring, expected_bin, tensor, simplified, matrix, id = parts
            circuit_filename = f"circuit_{id}.png"
            img_path = output_path / filename
            circuit_path = output_path / circuit_filename
            if not added_ham:
                ham_block = f"""
                <div class=\"section\">
                  <div class=\"matrix-container\">
                    <h3>Hamiltonian used for this simulation</h3>
                    <div class=\"item\">
                      <h4>Tensor Product Form:</h4>
                      <p>$$ {tensor} $$</p>
                    </div>
                    <div class=\"item\">
                      <h4>Matrix Form:</h4>
                      <p>$$ {matrix} $$</p>
                    </div>
                  </div>
                </div>
                """
                added_ham = True
            exact_block += f"""
            <div class=\"section\">
              <div class=\"container\">
                <div class=\"item\">
                  <h3>Time: {t} | Ancilla: {ancilla}</h3>
                  <p>Expected Binary: {expected_bin}</p>
                  <p>Measured: {bitstring}</p>
                </div>
                <div class=\"item\">
                  <img src=\"{circuit_path}\" alt=\"Circuit Diagram\">
                  <div class=\"label\">Quantum Circuit Diagram</div>
                </div>
                <div class=\"item\">
                  <img src=\"{filename}\" alt=\"Histogram\">
                  <div class=\"label\">Results for Exact Phase Estimation</div>
                </div>
              </div>
            </div>
            """
    body += ham_block + exact_block

    # Approximate Eigenvalue Estimation Section
    body += r"""
    <div class=\"section\">
      <h2>Approximate Eigenvalue Estimation of Hermitian Matrices</h2>
      <p>
      We aim to estimate the eigenvalues of a Hermitian matrix \( H \) using QPE.
      $$ e^{iHt} | \lambda \rangle = e^{2 \pi i \varphi} | \lambda \rangle $$
      $$ \lambda = \frac{2 \pi \varphi}{t} $$
      </p>
      <hr>
    """
    approx_block = ""
    added_ham2 = False
    for line in lines:
        parts = line.strip().split(",")
        if parts[0] == "general case":
            (_, filename, time, shots, ancilla, bitstring, est_phase,
             exp_phase, est_energy, ground_energy,
             tensor, simplified, matrix, expected_bitstring, id) = parts
            circuit_filename = f"circuit_{id}.png"
            unit_circle_filename = f"unit_circle_{id}.png"
            circuit_path = output_path / circuit_filename
            unit_circle_path = output_path / unit_circle_filename
            if not added_ham2:
                approx_block += f"""
                <div class=\"section\">
                  <div class=\"matrix-container\">
                    <h3>Hamiltonian used for this simulation</h3>
                    <div class=\"item\">
                      <h4>Tensor Product Form:</h4>
                      <p>$$ {tensor} $$</p>
                    </div>
                    <div class=\"item\">
                      <h4>Matrix Form:</h4>
                      <p>$$ {matrix} $$</p>
                    </div>
                  </div>
                </div>
                """
                added_ham2 = True
            approx_block += f"""
            <div class=\"section\">
              <div class=\"container\">
                <div class=\"item\">
                  <h3>Time: {time} | Shots: {shots} | Ancilla: {ancilla}</h3>
                  <p>Exact energy (\(\lambda\)): {round(safe_float(ground_energy), 5)}</p>
                  <p>Exact expected phase ({expression_for_expected_phase}): {round(safe_float(exp_phase), 5)}</p>
                  <p>Expected bitstring: ({expected_bitstring})</p>

                  <p>Most common measured bitstring (\(k\)): {bitstring}</p>
                  <p>Phase ({expression_for_calculated_phase_from_bitstring}): {round(safe_float(est_phase), 5)}</p>
                  <p>Estimated Energy ({expression_for_calculated_energy_from_phase}): {round(safe_float(est_energy), 5)}</p>
                </div>
                <div class=\"item\">
                  <img src=\"{circuit_path}\" alt=\"Circuit Diagram\">
                  <div class=\"label\">Quantum Circuit Diagram</div>
                </div>
                <div class=\"item\">
                  <img src=\"{filename}\" alt=\"Histogram\">
                  <div class=\"label\">Results for {shots} shots</div>
                </div>
                <div class=\"item\">
                  <img src=\"{unit_circle_path}\" alt=\"Unit Circle\">
                  <div class=\"label\">Complex Unit Circle</div>
                </div>
              </div>
            </div>
            <hr>
            """
    body += approx_block

    # qDRIFT Section
    body += r"""
    <div class=\"section\">
        <h2>qDRIFT Simulation</h2>
    """
    qdrift_block = ""
    added_ham3 = False
    for line in lines:
        parts = line.strip().split(",")
        if parts[0] == "qdrift general case":
            (_, filename, time, shots, ancilla, bitstring, est_phase,
             exp_phase, est_energy, ground_energy,
             tensor, simplified, matrix, expected_bitstring, id) = parts
            circuit_filename = f"circuit_{id}.png"
            unit_circle_filename = f"unit_circle_{id}.png"
            circuit_path = output_path / circuit_filename
            unit_circle_path = output_path / unit_circle_filename
            if not added_ham3:
                qdrift_block += f"""
                <div class=\"section\">
                  <div class=\"matrix-container\">
                    <h3>Hamiltonian used for this simulation</h3>
                    <div class=\"item\">
                      <h4>Tensor Product Form:</h4>
                      <p>$$ {tensor} $$</p>
                    </div>
                    <div class=\"item\">
                      <h4>Matrix Form:</h4>
                      <p>$$ {matrix} $$</p>
                    </div>
                  </div>
                </div>
                """
                added_ham3 = True
            qdrift_block += f"""
            <div class=\"section\">
              <div class=\"container\">
                <div class=\"item\">
                  <h3>Time: {time} | Shots: {shots} | Ancilla: {ancilla}</h3>
                  <p>Exact energy (\(\lambda\)): {round(safe_float(ground_energy), 5)}</p>
                  <p>Exact expected phase ({expression_for_expected_phase}): {round(safe_float(exp_phase), 5)}</p>
                  <p>Expected bitstring: ({expected_bitstring})</p>

                  <p>Most common measured bitstring (\(k\)): {bitstring}</p>
                  <p>Phase ({expression_for_calculated_phase_from_bitstring}): {round(safe_float(est_phase), 5)}</p>
                  <p>Estimated Energy ({expression_for_calculated_energy_from_phase}): {round(safe_float(est_energy), 5)}</p>
                </div>
                <div class=\"item\">
                  <img src=\"{circuit_path}\" alt=\"Circuit Diagram\">
                  <div class=\"label\">Quantum Circuit Diagram</div>
                </div>
                <div class=\"item\">
                  <img src=\"{filename}\" alt=\"Histogram\">
                  <div class=\"label\">Results for {shots} shots</div>
                </div>
                <div class=\"item\">
                  <img src=\"{unit_circle_path}\" alt=\"Unit Circle\">
                  <div class=\"label\">Complex Unit Circle</div>
                </div> 
                </div>
            </div>
            <hr>
            """
    body += qdrift_block

    body += """</body></html>"""
    

    # Write to file
    with open(html_path, "w") as html_file:
        html_file.write(body)

    print(f"Report generated: {html_path.resolve()}")

     # PDF generation
    # ... [build HTML exactly as before and write to html_path] ...
    html_to_pdf_path = output_path / pdf_file  # e.g. 'qpe_output_report.pdf'
    asyncio.run(html_to_pdf(html_path, html_to_pdf_path))
    print(f"PDF report generated via Playwright: {html_to_pdf_path.resolve()}")


if __name__ == "__main__":
    generate_report(only_hamiltonian_simulation=True, pdf_file='qpe_output_report2.pdf')
