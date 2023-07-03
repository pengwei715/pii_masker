import click

@click.command()
@click.option(
    "--input_file", "-i", type=click.Path(exists=True), help="Input file path"
)
@click.option("--output_file", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--model",
    "-m",
    type=click.Choice(["spacy", "flair", "pattern", "whole"]),
    default="whole",
    help="Model type (predefined_model_with spacy or flair, pattern_recognizer or combine those toghether)",
)
@click.option(
    "--file_type",
    "-t",
    type=click.Choice(["html", "txt"]),
    default="txt",
    help="File type (txt, html)",
)
@click.option("--stats_report", "-s", is_flag=True, help="Generate stats report")
def run(input_file, output_file, file_type, model, stats_report):
    """Process the input file and generate the output file."""
    click.echo(
        f"Running with input file: {input_file}, output file: {output_file}, file type: {file_type} using model {model}"
    )

    # Perform the necessary operations based on the provided options
    # Replace the code below with your own logic
    if input_file and output_file:
        click.echo("Processing the file...")
        click.echo(f"Generating {file_type.upper()} file: {output_file}")
        if stats_report:
            click.echo("Generating stats report...")

    click.echo("Process completed.")




if __name__ == "__main__":
    run()
