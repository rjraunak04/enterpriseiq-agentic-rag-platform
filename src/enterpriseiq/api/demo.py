"""Recruiter-facing demo interface for EnterpriseIQ."""

from __future__ import annotations

from html import escape

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["demo"])


@router.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo(request: Request) -> HTMLResponse:
    """Serve a dependency-free interface for exercising the agent API."""

    config = request.app.state.api_config
    endpoint = f"{config.api_prefix}/agent"
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EnterpriseIQ Demo</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;background:#f6f7f9;color:#17202a}}
main{{background:white;padding:28px;border-radius:14px;box-shadow:0 8px 30px #0001}}
textarea{{width:100%;box-sizing:border-box;min-height:110px;padding:12px}}
button{{margin-top:10px;padding:10px 18px;cursor:pointer}}
pre{{white-space:pre-wrap;background:#111827;color:#e5e7eb;padding:16px;border-radius:8px;min-height:100px}}
small{{color:#667085}}
</style>
</head>
<body><main>
<h1>EnterpriseIQ</h1>
<p>Grounded enterprise knowledge retrieval with controlled agent tool execution.</p>
<textarea id="query" maxlength="{config.max_query_length}" placeholder="Ask an enterprise policy question..."></textarea>
<br><button id="run">Run grounded query</button>
<p><small>API: {escape(endpoint)} · Max query length: {config.max_query_length}</small></p>
<pre id="output">Ready.</pre>
</main>
<script>
const output=document.getElementById("output");
document.getElementById("run").addEventListener("click",async()=>{{
 const query=document.getElementById("query").value.trim();
 if(!query){{output.textContent="Enter a query.";return;}}
 output.textContent="Running...";
 try{{
  const response=await fetch({endpoint!r},{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{query}})}});
  const data=await response.json();
  output.textContent=JSON.stringify(data,null,2);
 }}catch(error){{output.textContent="Request failed. Check service availability.";}}
}});
</script>
</body></html>"""
    return HTMLResponse(html)
