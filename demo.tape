# Hotel Preference 20 Questions - interactive demo
# Render with: vhs demo.tape

Output demo.mp4

Set Shell "pwsh"
Set Width 1400
Set Height 900
Set FontSize 24
Set Padding 20
Set TypingSpeed 60ms
Set Theme "Catppuccin Mocha"

# Activate the project venv off-camera
Hide
Type `$env:Path = "$PWD\.venv\Scripts;$env:Path"; function prompt { '> ' }; Clear-Host`
Enter
Sleep 2s
Show

Type "python hotel_preference_demo.py --interactive"
Sleep 500ms
Enter
Sleep 5s

# Five questions; answer the beach/quiet (B) side each time
Type "B"
Sleep 400ms
Enter
Sleep 3.5s

Type "B"
Sleep 400ms
Enter
Sleep 3.5s

Type "B"
Sleep 400ms
Enter
Sleep 3.5s

Type "B"
Sleep 400ms
Enter
Sleep 3.5s

Type "B"
Sleep 400ms
Enter
Sleep 5s
