@echo off
REM uvを使用してPythonの仮想環境を作成し、パッケージをインストールするバッチファイル

echo === uvを使用して仮想環境を作成しています ===
uv venv -p 3.10

echo === 仮想環境をアクティベートしています ===
call .venv\Scripts\activate

echo === requirements.txtからパッケージをインストールしています ===
uv pip install -r requirements.txt

echo === インストールされたパッケージを表示します ===
uv pip list

echo === 完了しました ===
echo 仮想環境を終了するには 'deactivate' と入力してください

pause