# Configure Conda in PowerShell

These steps configure PowerShell so that Conda is available automatically and
the default Conda environment is active when a new terminal opens. A successful
setup displays `(base)` at the beginning of the PowerShell prompt.

## Prerequisite

Install a Conda distribution, such as Miniconda, Anaconda, or Miniforge.

## Initialize PowerShell

1. Open the prompt installed with your Conda distribution, such as **Anaconda
   Prompt** or **Miniconda Prompt**.
2. Initialize Conda for PowerShell:

   ```powershell
   conda init powershell
   ```

3. Enable automatic activation of Conda's default environment:

   ```powershell
   conda config --set auto_activate true
   ```

   If an older Conda version does not recognize `auto_activate`, use:

   ```powershell
   conda config --set auto_activate_base true
   ```

4. Close all PowerShell windows, then open a new one.

The prompt should now begin with `(base)`:

```text
(base) PS C:\Users\username>
```

## Create the project environment

From the repository root, create the environment defined in `env.yml`:

```powershell
conda env create --file env.yml
conda activate apt
```

If the `apt` environment already exists, update it instead:

```powershell
conda env update --file env.yml --prune
conda activate apt
```

## Troubleshooting

### PowerShell blocks the profile script

If PowerShell reports that scripts are disabled, open a normal, non-administrator
PowerShell window and run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then run `conda init powershell` again from the Conda prompt, close all
PowerShell windows, and reopen PowerShell.

### Inspect the current configuration

Use these commands to locate and inspect the PowerShell profile and Conda
configuration:

```powershell
$PROFILE
Get-Content $PROFILE
conda config --show-sources
conda config --show auto_activate
```

`conda init powershell` adds a Conda-managed initialization block to the current
user's PowerShell profile. The automatic-activation setting causes Conda's
default environment (normally `base`) to activate when that profile loads.

The directory shown after `PS` in the prompt is unrelated to Conda. For example,
starting in `C:\WINDOWS\system32` is normally determined by the shortcut or
process used to launch PowerShell.

For more information, see the official
[`conda init` documentation](https://docs.conda.io/projects/conda/en/stable/commands/init.html).
