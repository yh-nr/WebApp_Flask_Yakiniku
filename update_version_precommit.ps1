$VERSION_FILE = "src\.version"
$DATE_FORMAT = Get-Date -UFormat "%y%m%d"

# Extract the current patch number
$CURRENT_VERSION = Get-Content $VERSION_FILE
$CURRENT_PATCH = ($CURRENT_VERSION -split "-")[1]

# Increment the patch number
$NEW_PATCH = [int]$CURRENT_PATCH + 1

"$DATE_FORMAT-$NEW_PATCH" | Out-File $VERSION_FILE -Encoding ascii

# Add the updated .version file to the commit
git add $VERSION_FILE

