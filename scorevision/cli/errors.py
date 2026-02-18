class CLIError(Exception):
    pass


class ConfigError(CLIError):
    pass


class DockerBuildError(CLIError):
    pass


class DockerPushError(CLIError):
    pass


class DockerRunError(CLIError):
    pass


class ChainCommitError(CLIError):
    pass


