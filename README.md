# ScoreVision

Mine vision.

[score Discord](https://discord.gg/eh8ja2t92y)

# Setup 
## Bittensor
Get a Bittensor wallet

### Install bittensor cli
```bash
pip install bittensor-cli
```

### Create a coldkey (your main wallet)
```bash
btcli wallet new_coldkey --n_words 24 --wallet.name my-wallet
```

### Create a hotkey (for signing transactions)

```bash
btcli wallet new_hotkey --wallet.name my-wallet --n_words 24 --wallet.hotkey my-hotkey
```

### Update your .env file
```bash
cat >> .env << 'EOF'
BITTENSOR_WALLET_COLD=coldkey
BITTENSOR_WALLET_HOT=my-hotkey
BITTENSOR_WALLET_PATH=path-~/.bittensor/wallets/my-wallet/hotkeys/my-hotkey
EOF
```

## Chutes 

### Install chutes cli
`pip install -U chutes`

### Register your account
`chutes register`

Follow the interactive prompts to:

- Enter your desired username
- Select your Bittensor wallet
- Choose your hotkey
- Complete the registration process (note your unique fingerprint)

### Create an API key
- Log into chutes website (via your fingerprint)
- Create an API token 

### Update your .env file
```bash
cat >> .env << 'EOF'
CHUTES_USERNAME=your-username
CHUTES_API_KEY=your-api-key
EOF
```


## Huggingface
- Create a Huggingface account and sign in
- Create a token

### Update your .env file
```bash
cat >> .env << 'EOF'
HUGGINGFACE_USERNAME=your-username
HUGGINGFACE_API_KEY=your-api-key
EOF
```

## Cloudflare R2
- Create a cloudflare account and sing in 
- Create token

### Update your .env file
```bash
cat >> .env << 'EOF'
R2_BUCKET=your-bucket
R2_ACCOUNT_ID=your-account-id
R2_WRITE_ACCESS_KEY_ID=your-key-id
R2_WRITE_SECRET_ACCESS_KEY=your-key
EOF
```

# Scorevision CLI
![](images/sv-cli.png)

# Validators

(Recommended): Run the validator with docker.
```bash
docker compose down && docker compose pull 
docker compose up --build -d && docker compose logs -f
```

(Optional): Run the validator locally
```bash
sv -vv validate
sv -vv runner
sv -vv signer
```



# Miners

0. Install Scorevision CLI (uv)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# setup
uv venv && source .venv/bin/activate
uv sync  # install deps

# verify installation
sv
```

1. Register with the Subnet 
Register your miner to ScoreVision (S44).
```bash
btcli subnet register --wallet.name <your cold> --wallet.hotkey <your hot>
```

2. Upgrade Chutes to a Developer-Enable Account
Miners need a chutes developer account ( `chutes.ai` ). 

IMPORTANT: you require a ***developer enabled account*** on Chutes to mine. Normal API keys cannot deploy chutes right now.

3. Train a model
```bash
... magic ML stuff ...
```

4. Modify how the model will be deployed on Chutes
Before you push your model, you can customise how it is loaded in on chutes and handles predictions

- `scorevision/chute_template/setup.py`
- `scorevision/chute_template/load.py`
- `scorevision/chute_template/predict.py`

It is highly recommended to build the chute locally after any changes to check for errors. Steps to build locally:
- (optional) ssh onto a machine with the specs matching your requirements (e.g. GPU, etc)
- ensure your `~/.chutes/config.ini` file is present on the machine (generated automatically when you register with chutes)
- install docker and chutes 
- generate the python script containing your chute called "my_chutes.py" via `sv -v generate-chute-script --revision your-hf-sha`
- run `chutes build my_chutes:chute --local --public`
- `docker images`
- run the image you just built and enter it `docker run -p 8000:8000 -it <image-name> /bin/bash` 
- when inside the container: `export CHUTES_EXECUTION_CONTEXT=REMOTE` and run `chutes run my_chute:chute --dev --debug`
- query the endpoints from outside the container

5. Push the model to your miner
Once you are happy with the changes, push your model to Huggingface Hub and then deploy it to Chutes and onto Bittensor with the following command:

```bash
sv -vv push --model_path <i.e. ./my_model>
```

Note the hugginface repo revision and chute slug from the logs.  If you missed it you can get the revision directly from your Hf account and you can use this to get your chute-slug and ID:
```bash
sv -v chute-slug --revision your-huggingface-repo-sha
```

(Be careful, you only have a limited number of uploads per 24hours)

6. Test it live
Soon your model will be hot on chutes. You can check that using 
```bash
chutes chutes list
```

You can test the /health and /predict endpoints using a fake challenge payload like so:

```bash 
sv -v ping-chute --revision your-huggingface-repo-sha
``` 

You can also run the full validator scoring mechanism on the model with a fake challenge payload like so:
```bash 
sv -vv test-chute --revision your-huggingface-repo-sha 
```

If you are finding problems with your live chute, you can view its logs like so:
- log into chutes via the browser (use your fingerprint)
- find the chute "My Chutes"
- go to the "Statistics" tab
- note down the instance-id
- query the logs via the api: `curl -XGET https://api.chutes.ai/instances/<CHUTE-INSTANCE-ID>/logs -H "Authorization: <CHUTES-API-KEY>"`

7. Delete old models (optional)
You can remove an old version of your model from chutes if desired
```bash
sv -v delete-chute --revision your-old-huggingface-repo-sha 
```




