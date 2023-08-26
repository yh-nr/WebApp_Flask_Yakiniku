# .git/hooks/pre-commitから呼び出す

# バージョン情報を格納するファイルのパスを設定
$VERSION_FILE = "src\.version"

# 現在の日付をYYMMDD形式で取得
$DATE_FORMAT = Get-Date -UFormat "%y%m%d"

# .version ファイルから現在のバージョン番号を取得
$CURRENT_VERSION = Get-Content $VERSION_FILE

# 現在のパッチ番号を取得（"-"で分割して2つ目の値を取る）
$CURRENT_PATCH = ($CURRENT_VERSION -split "-")[1]

# パッチ番号を1増加させる
$NEW_PATCH = [int]$CURRENT_PATCH + 1

# 新しいバージョン番号を.versionファイルに書き込む
"$DATE_FORMAT-$NEW_PATCH" | Out-File $VERSION_FILE -Encoding ascii

# 変更した.versionファイルをgitのステージングエリアに追加
git add $VERSION_FILE
