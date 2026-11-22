Set-Location "D:\Code\law-wiki-lab"
git add -A
Set-Content -Path tmp_commit_msg.txt -Value "第14层：知识一致性维护，笔记记依据条文@指纹，sync 自动重生成过期笔记" -Encoding utf8
git commit -F tmp_commit_msg.txt
if (Test-Path tmp_commit_msg.txt) { Remove-Item tmp_commit_msg.txt }
git push
git log --oneline -3