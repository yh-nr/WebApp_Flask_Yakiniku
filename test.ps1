$VERSION_FILE = ".\src\.version"
$DATE_FORMAT = Get-Date -UFormat "%y%m%d"

# Check if the .version file has been modified
if (git diff --cached --name-only -z | %{ $_ -split '\0' } | Where-Object { $_ -eq $VERSION_FILE }) {
    # Check if the file exists
    if (-not (Test-Path $VERSION_FILE)) {
        "$DATE_FORMAT-1" | Out-File $VERSION_FILE -Encoding ascii
        exit 0
    }

    # Extract the current patch number
    $CURRENT_VERSION = Get-Content $VERSION_FILE
    $CURRENT_PATCH = ($CURRENT_VERSION -split "-")[1]

    # Increment the patch number
    $NEW_PATCH = [int]$CURRENT_PATCH + 1

    "$DATE_FORMAT-$NEW_PATCH" | Out-File $VERSION_FILE -Encoding ascii

    # Add the updated .version file to the commit
    git add $VERSION_FILE
}
