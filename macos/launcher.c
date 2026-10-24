// Native launcher for Hush.app.
//
// Embeds the Python interpreter instead of exec'ing it. An exec'd Python
// process belongs to Python.app, so the Dock and menu bar would say "Python"
// with Python's icon. Embedding keeps the process inside Hush.app.
//
// HUSH_REPO_DIR is baked in at compile time by build_app.sh. The app runs
// from the repo directory because hush resolves hush.db, models/ and
// screenshots/ relative to the working directory.

#include <Python.h>

#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef HUSH_REPO_DIR
#error "HUSH_REPO_DIR must be defined at compile time"
#endif

static const char VENV_PYTHON[] = HUSH_REPO_DIR "/.venv/bin/python";
static const char LOG_PATH_IN_HOME[] = "Library/Logs/Hush.log";

// A Finder-launched app has no terminal, so send stdout/stderr to a log
// file. It is truncated on each launch so it holds the most recent run.
static void redirect_output_to_log(void) {
    const char *home = getenv("HOME");
    if (home == NULL) {
        return;
    }
    char log_path[PATH_MAX];
    snprintf(log_path, sizeof log_path, "%s/%s", home, LOG_PATH_IN_HOME);
    int fd = open(log_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return;
    }
    dup2(fd, STDOUT_FILENO);
    dup2(fd, STDERR_FILENO);
    close(fd);
}

int main(int argc, char **argv) {
    (void)argc;
    redirect_output_to_log();

    if (chdir(HUSH_REPO_DIR) != 0) {
        perror("Hush: cannot chdir to " HUSH_REPO_DIR);
        return EXIT_FAILURE;
    }

    // Framework builds of Python read this to find the interpreter that was
    // "really" launched; pointing it at the venv's python makes the venv's
    // site-packages (and sys.executable) apply to the embedded interpreter.
    setenv("__PYVENV_LAUNCHER__", VENV_PYTHON, 1);

    char *python_argv[] = {argv[0], "-m", "hush.v2app", NULL};
    return Py_BytesMain(3, python_argv);
}
