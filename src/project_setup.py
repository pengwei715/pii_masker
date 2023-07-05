import mlrun
import importlib

IMAGE_REQUIREMENTS = [
    "pandas",
    "streamlit",
    "presidio-anonymizer",
    "presidio-analyzer",
    "torch",
    "flair@git+https://github.com/flairNLP/flair.git@d4ed67bf663e4066517f00397412510d90043653",
    "st-annotated-text",
    "https://huggingface.co/beki/en_spacy_pii_distilbert/resolve/main/en_spacy_pii_distilbert-any-py3-none-any.whl",
]


def assert_build():
    for module_name in IMAGE_REQUIREMENTS:
        module = importlib.import_module(module_name)
        print(f"Successfully imported {module_name}.")


def create_and_set_project(
    git_source: str,
    name: str = "pii",
    default_image: str = None,
    default_base_image: str = "mlrun/ml-models",
    user_project: bool = True,
):
    """
    Creating the project for this demo.
    :param git_source:              the git source of the project.
    :param name:                    project name
    :param default_image:           the default image of the project
    :param user_project:            whether to add username to the project name

    :returns: a fully prepared project for this demo.
    """
    # Get / Create a project from the MLRun DB:
    project = mlrun.get_or_create_project(
        name=name, context="./", user_project=user_project
    )

    # Set or build the default image:
    if project.default_image is None:
        if default_image is None:
            print("Building image for the demo:")
            image_builder = project.set_function(
                "project_setup.py",
                name="image-builder",
                handler="assert_build",
                kind="job",
                image=default_base_image,
            )
            build_status = project.build_function(
                function=image_builder,
                base_image=default_base_image,
                requirements=IMAGE_REQUIREMENTS,
            )
            default_image = build_status.outputs["image"]
        project.set_default_image(default_image)

    # Set the project git source:
    project.set_source(git_source, pull_at_runtime=True)

    # Set the data collection function:
    project.set_function(
        "process.py",
        name="process",
        image=default_image,
        kind="job",
        handler="process",
        with_repo=True,
        requirements=IMAGE_REQUIREMENTS,
    )

    # Save and return the project:
    project.save()
    return project


if __name__ == "__main__":
    proj = get_or_create_project(
        git_source="https://github.com/pengwei715/pii_masker",
        name="pii",
        default_image="mlrun/ml-models",
        user_project=True,
    )

    proj.run_function(
        "process",
        handler="process",
        params={
            "input_file": "data/pii.txt",
            "output_file": "data/pii_output.txt",
            "model": "whole",
            "stats_report": True,
        },
    )
