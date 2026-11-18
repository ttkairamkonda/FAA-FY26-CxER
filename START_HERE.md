# Start here

For a local machine or Lambda server **without Docker or sudo**, follow
[LOCAL_STUDIO.md](LOCAL_STUDIO.md). It includes dependency setup, GPU selection,
local judges, ATC/FAA results and batch exports.

The guide below is the separate Docker/AWS deployment option.

# FAA CxER Studio: setup guide (no coding needed)

This guide takes you from nothing to the running app in your web browser. You only
need **PowerShell** (built into Windows) and an **AWS account**. Every command is
written out; copy and paste them **one box at a time**.

**What the app does:** you upload air-traffic-control audio. The app transcribes it with
several speech-recognition models and shows how accurate each one is (WER). If the server
has big GPUs, it also asks an AI judge whether each transcription contains a
**safety-critical error** (wrong callsign, altitude, runway, and so on). That score is
called CxER.

> **Two windows, two places.** You'll use two PowerShell windows:
> - A prompt that starts with `PS C:\Users\...>` is **your computer**.
> - A prompt that starts with `ubuntu@ip-...` is **the AWS server**.
>
> Each step below says where to type.

---

## Step 1. Choose your server size

| | **Full version** (with the CxER safety judge) | **Basic version** (no judge) |
|---|---|---|
| What works | Everything: transcription, accuracy, safety verdicts | Transcription and accuracy only. The app shows a note that the judge needs a GPU. |
| AWS instance type | `g6e.12xlarge` (4 GPUs, 192 GB GPU memory) | `m7i.xlarge` (no GPU, 16 GB memory) |
| Approx. cost | about **$10 per hour** while running | about **$0.20 per hour** while running |
| Disk size to choose | **300 GB** | **50 GB** |

You don't need to decide anything else. The setup script detects the hardware and starts
the right version by itself.

**Full version only: ask AWS for GPU access first.** New accounts get 0 GPUs by default.
1. In the AWS console search bar, type **Service Quotas** and open it.
2. Click **Amazon Elastic Compute Cloud (Amazon EC2)**.
3. Search for **Running On-Demand G and VT instances**, click it, then **Request increase at account level**.
4. Enter **48** and submit. Approval can take from minutes to a couple of days, and you'll get an email.

## Step 2. Get a Hugging Face token (full version only)

The safety judge is the Llama-3.3-70B model, and it's free but needs an account.
1. Create a free account at https://huggingface.co.
2. Open https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct, fill in the short form,
   and accept the license. Wait until the page says you have access.
3. Click your profile picture, then **Settings → Access Tokens → Create new token**. Choose **Read**, then create it.
4. Copy the token (it starts with `hf_`) and keep it somewhere safe. You'll paste it in step 6.

## Step 3. Create the server

In the AWS console:
1. At the top right, choose the region **US East (N. Virginia) us-east-1**.
2. Search for **EC2**, open it, and click **Launch instance**.
3. **Name:** `cxer-studio`
4. **Application and OS Images:** click **Browse more AMIs**, then search and pick:
   - Full version: **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**
   - Basic version: **Ubuntu Server 24.04 LTS**
5. **Instance type:** `g6e.12xlarge` (full) or `m7i.xlarge` (basic). See step 1.
6. **Key pair:** click **Create new key pair**. Name it `cxer-key`, type **RSA**, format **.pem**,
   then click **Create**. A file `cxer-key.pem` downloads to your **Downloads** folder. Keep it,
   because you can't download it again.
7. **Network settings:** keep "Allow SSH traffic" checked and change **Anywhere** to **My IP**.
8. **Configure storage:** change the size to **300** GiB (full) or **50** GiB (basic).
9. Click **Launch instance**, then **View all instances**. Wait until the status is
   **Running**, then click the instance and copy its **Public IPv4 address** (for example `3.144.9.88`).

## Step 4. Prepare the key (on your computer, one time)

Open **PowerShell**: press the Windows key, type `PowerShell`, and press Enter. Paste:

```
icacls "$HOME\Downloads\cxer-key.pem" /inheritance:r
```
```
icacls "$HOME\Downloads\cxer-key.pem" /grant:r "$($env:USERNAME):(R)"
```

This makes the key private; SSH refuses keys that other accounts can read.

## Step 5. Connect to the server (on your computer)

Replace `3.144.9.88` with **your** server's address from step 3:

```
ssh -i "$HOME\Downloads\cxer-key.pem" ubuntu@3.144.9.88
```

The first time, it asks *"Are you sure you want to continue connecting?"*. Type `yes` and
press Enter. The prompt changes to `ubuntu@ip-...`, which means you're now on the server.

## Step 6. Download and start the app (on the server)

```
git clone https://github.com/ttkairamkonda/FAA-FY26-CxER.git
```
If GitHub asks for a username and password, the repository is private. Use your GitHub username,
and for the password a **personal access token** (GitHub → Settings → Developer settings →
Personal access tokens). Ask the project owner for access if needed.

```
cd FAA-FY26-CxER
```
```
bash studio.sh start
```

The script now:
- checks the server (GPUs, memory, disk) and prints what it found;
- installs anything missing;
- asks for your **Hugging Face token** (full version only). Paste it and press Enter.
  Nothing appears on screen while you paste; that's normal;
- builds and starts the app. The first time takes **5 to 10 minutes**.

At the end it prints a box with two instructions. Leave this window open.

## Step 7. Open the app (on your computer)

Open a **second** PowerShell window and run (with your server address):

```
ssh -i "$HOME\Downloads\cxer-key.pem" -L 8501:localhost:8501 ubuntu@3.144.9.88
```

**Leave this window open** while you use the app. It's a private, encrypted link to the server.

Now open your web browser at **http://localhost:8501**. The app appears.

**Full version:** the safety judge loads in the background, and the first time takes
**15 to 40 minutes** (it downloads about 140 GB). Until then the app shows "CxER judge
unavailable". To check progress, run this in the server window:
```
bash studio.sh status
```
When it says **CxER judge: READY**, click **Re-check judge** in the app's left panel.

## Step 8. Use the app

1. On the **Analyze audio** tab, choose **Upload files** and drop in your audio files
   (wav, mp3, flac, m4a, ogg). You can also choose **Use a bundled ATC clip** to try a sample.
2. For each file, type what was actually said in **Reference transcript**. That's needed for
   the accuracy score and the safety verdict.
3. In the left panel, pick the models to compare (the default pair is fine).
4. Click **Run analysis**. Each model's transcription appears with:
   - **WER**: the share of words it got wrong (lower is better);
   - a green **Equivalent** or red **Critical error** label (full version), with a table of
     what was misheard and how risky it is.
5. **Download CSV / JSON** saves the results. The **Benchmark results** tab shows the project's
   overall findings.

The first analysis is slow, because the speech models download about 1.5 GB each and load into memory.
Later runs are much faster.

## Step 9. When you're done (important: save money)

In the server window:
```
bash studio.sh stop
```
Then, in the AWS console under **EC2 → Instances**, select the server and choose
**Instance state → Stop instance**. **A running server is billed every hour**, even if nobody uses it.

**Next time:** select the server, choose **Instance state → Start instance**, copy its **new**
Public IPv4 address (it changes), and repeat steps 5, 7 and this one command on the server:
```
cd FAA-FY26-CxER && bash studio.sh start
```
It starts in about a minute. (Full version: the judge reloads its model, about 5 to 10 minutes.)

When you no longer need it at all: **Instance state → Terminate instance**. This deletes the
server and stops all charges.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `Permission denied (publickey)` | Check the key path in your command and the server address. Use `ubuntu@` (not `ec2-user@`). Run step 4 again. |
| `UNPROTECTED PRIVATE KEY FILE` / `bad permissions` | Run step 4 again. |
| `Connection timed out` | The server is stopped, or its address changed (copy the new one), or your own IP changed. In that case, open the instance's **Security** tab → security group → **Edit inbound rules** and set the SSH rule's source to **My IP** again. |
| Launching fails with *"You have requested more vCPU capacity than your current vCPU limit"* | Your GPU quota isn't approved yet (step 1). Use the basic version meanwhile. |
| The browser says "This site can't be reached" | The second PowerShell window from step 7 must be open and connected. |
| App shows "ASR failed", and Error details mentions memory | The server is too small. Use `m7i.xlarge` or bigger. |
| Script says "too little free disk" | In EC2 → **Volumes**, select the server's volume → **Modify** → make it bigger. Then, on the server, run `sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1`, then `bash studio.sh start` again. |
| `CxER judge: stopped` | Run `bash studio.sh logs`. A 401 or 403 means the Hugging Face token or license (step 2) isn't right. Delete the saved token with `rm .env`, then run `bash studio.sh start` and paste it again. |
| Anything else | Run `bash studio.sh logs` on the server and send the output to the project team. |

### Useful commands (on the server, inside the `FAA-FY26-CxER` folder)

| Command | What it does |
|---|---|
| `bash studio.sh start` | Start or restart everything |
| `bash studio.sh status` | Is the app running? Is the judge ready? |
| `bash studio.sh logs` | Show what the app is doing (press Ctrl+C to leave) |
| `bash studio.sh stop` | Stop the app |
| `exit` | Disconnect from the server |
