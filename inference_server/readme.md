## Running the Inference Server

To run the inference server successfully, please follow these steps:

1. **Install Docker Desktop**  
   Make sure Docker Desktop is installed on your machine.

2. **Start Docker Desktop**  
   Docker Desktop must be **running** before continuing.

3. **Mount Your Hugging Face Hub Folder**  
   Adjust line 3 (e.g. `YOUR_PATH="D:/models/Huggingface/"`) in `start_server.sh` to point to your local systems huggingface cache directory.

4. **Set Your Hugging Face Token**  
   Replace the placeholder Hugging Face token with your own valid token.

5. **Check Port Availability**  
   Ensure that **no other application is using port `30000`** on `localhost`.

6. **Verify the Model Path**  
   Confirm that the model path exists—either locally or on Hugging Face.

7. **Check RAM/VRAM Requirements**  
   Ensure the model you plan to use fits within your available system RAM and/or GPU VRAM.

8. **Ensure Sufficient Disk Space**  
   Your system should have enough storage to download:
   - The **SGLang Docker container** (~10 GB)
   - The **model** itself (~16 GB for Llama 3.1 8B Instruct; size varies by model)

---

💡 **Tip:** To run different configurations with the same model, refer to the official documentation:  
[https://docs.sglang.ai/backend/server_arguments.html](https://docs.sglang.ai/backend/server_arguments.html)

---

✅ **Once all the conditions are satisfied**, start the inference server by executing the `start_server.sh` script in a **Linux terminal** or **Git Bash** on Windows.

⚠️ **Note:**  
First-time setup may take a while, as it requires downloading several large files to your local system.

**Check if service is running**
You can check if your service is running using `http://localhost:30000/get_model_info`. 
Additionally, the console in which you ran `start_server.sh` should show logs similar to 
```
[2025-07-14 05:48:43] Load weight end. type=LlamaForCausalLM, dtype=torch.bfloat16, avail mem=7.37 GB, mem usage=15.02 GB.
[2025-07-14 05:48:43] KV Cache is allocated. #tokens: 38141, K size: 2.33 GB, V size: 2.33 GB
[2025-07-14 05:48:43] Memory pool end. avail mem=1.46 GB
[2025-07-14 05:48:44] Capture cuda graph begin. This can take up to several minutes. avail mem=0.93 GB
[2025-07-14 05:48:44] Capture cuda graph bs [1, 2, 4, 8]
Capturing batches (bs=1 avail_mem=0.86 GB): 100%|██████████| 4/4 [00:18<00:00,  4.51s/it]
[2025-07-14 05:49:02] Capture cuda graph end. Time elapsed: 18.07 s. mem usage=0.11 GB. avail mem=0.83 GB.
[2025-07-14 05:49:02] max_total_num_tokens=38141, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=2048, context_len=131072, available_gpu_mem=0.83 GB
[2025-07-14 05:49:03] INFO:     Started server process [1]
[2025-07-14 05:49:03] INFO:     Waiting for application startup.
[2025-07-14 05:49:03] INFO:     Application startup complete.
[2025-07-14 05:49:03] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

## Performing Inference

Example script:

```
from utils import ComputeClient

client = ComputeClient()
model_response = client.call_llm('Das ist ein Beispiel System Prompt', 'Das ist ein Beispiel prompt')
```

The compute client can also take a list of prompts for batched inference and either a list of system prompts or one system prompt as input.
Additionally, it supports all SGLang native inference parameters, you can read more on them at https://docs.sglang.ai/backend/sampling_params.html



