# NLF to SMPLer-X Replacement Plan (Non-Disruptive)

This document details how to replace **SMPLer-X** body model initialization with **NLF (Neural Localizer Fields)** initialization within the DexAvatar fitting pipeline, without modifying the codebase or affecting other running methods.

---

## 1. Context and Mechanics

During Phase 2 and 3 of the DexAvatar pipeline (defined in scripts like `continue_with_shared.sh`), the fitting script `smplifyx/main.py` is invoked with a target directory `--data_folder <method_sign_dir>`. 

By default, the Python fitting code looks for initialization parameters inside:
```
<data_folder>/smplerx/smplx/{frame_name}.pkl
```
This path is controlled by the configuration variable `smplx_init_dir` (default value is `"smplerx/smplx"`).

To use NLF instead of SMPLer-X without breaking or modifying Python code, you can use either **Approach A (Symlink redirection)** or **Approach B (Config-level YAML separation)**.

---

## 2. Approach A: Symlink Redirection (Zero Code/Config Changes)

This approach redirects file-system queries for `smplerx` to the `nlf` output folder. Since other methods rely on the symlink structure created in `outputs/shared/`, redirecting it here ensures all downstream methods automatically consume NLF without modifying any code or configurations.

### Steps to Replace SMPLer-X with NLF:
Run the following commands in the terminal to replace the shared `smplerx` symlink with a pointer to `nlf`:

```bash
# 1. Navigate to project root
cd /home/haipd/DexAvatar

# 2. Redirect shared smplerx folders to nlf for all processed signs
for d in outputs/shared/*/; do
    if [ -d "$d/nlf" ]; then
        rm -rf "$d/smplerx"
        ln -s nlf "$d/smplerx"
        echo "Redirected $(basename $d) to NLF"
    fi
done

# 3. Refresh symlinks in method-specific output directories
bash scripts/link_shared_to_methods.sh
```

### Steps to Revert back to original SMPLer-X:
If you need to restore the original SMPLer-X initializations, delete the symlinks and restore the original pointers:

```bash
for d in outputs/shared/*/; do
    # Check if smplerx is currently pointing to nlf
    if [ -h "$d/smplerx" ] && [ "$(readlink "$d/smplerx")" = "nlf" ]; then
        rm "$d/smplerx"
        # Re-create the folder or standard symlink (depending on your setup)
        # Note: If you unzipped original SMPLer-X predictions elsewhere, restore that link.
    fi
done
```

---

## 3. Approach B: Config-Level YAML Isolation (Method-Specific)

If you only want to replace SMPLer-X with NLF for a *specific* method/experiment (e.g. only for a new method you are developing) while keeping the original methods completely untouched, you can use a custom configuration file.

### Step 1: Create a Custom Config File
Create a copy of your configuration file, e.g., `dexavatar_fitting/cfg_files/fit_smplx_vposer_x_nlf.yaml`, and add the `smplx_init_dir` property:

```yaml
# fit_smplx_vposer_x_nlf.yaml
# ... (copy original config contents) ...

# Override the body initialization directory
smplx_init_dir: "nlf/smplx"
```

### Step 2: Establish the NLF Symlink in Output Directories
Because the default link script `link_shared_to_methods.sh` only links `smplerx` and `hamer` to the method output folder, you must link the `nlf` directory manually into the method's folder:

```bash
# Example for a method output 'outputs/method_custom' and sign 'Akzeptieren'
METHOD_DIR="/home/haipd/DexAvatar/outputs/method_custom"
SIGN_NAME="Akzeptieren"

mkdir -p "${METHOD_DIR}/${SIGN_NAME}"
ln -sf "../../shared/${SIGN_NAME}/nlf" "${METHOD_DIR}/${SIGN_NAME}/nlf"
```

### Step 3: Run Fitting with the Custom Config
Specify the new configuration YAML file when running the fitting runner script:

```bash
python smplifyx/main.py \
    --config cfg_files/fit_smplx_vposer_x_nlf.yaml \
    --data_folder outputs/method_custom/Akzeptieren \
    --output_folder outputs/method_custom/Akzeptieren/smplifyx \
    --img_folder data/frames/Akzeptieren \
    --model_folder ../SMPLer-X/common/utils/human_model_files
```

---

## 4. Verification

After launching the fitting pipeline with NLF replacements, you can check that it has integrated successfully by inspecting the generated output meshes or log files:

1. **Verify Log Files (`pipeline.log` / terminal outputs):**
   Open the log file for any sign and verify that the fitting initialization is successfully reading from the NLF path:
   ```bash
   grep -i "nlf" outputs/method_nlf_wilor/Akzeptieren/pipeline.log
   ```
2. **Verify Mesh Generation:**
   Check that meshes are being written correctly under `outputs/method_nlf_wilor/Akzeptieren/smplifyx/meshes/*.obj`. Their shape and alignment will reflect NLF body bounds combined with WiLoR hands.
