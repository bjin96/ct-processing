"""Helper functions for calling the command line utilities."""
import shlex
import subprocess


def call_command_line_verbose(
        commandline: str,
) -> None:
    """
    Call a program on the command line. Can throw a subprocess.CalledProcessError if there is a problem with the call.
    """
    command = shlex.split(commandline)
    print(f'Calling {" ".join(command)}')
    completed_process = subprocess.run(command, capture_output=True)
    print(f'{completed_process.stdout=}, {completed_process.stderr=}')
