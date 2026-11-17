; The Windows installer of the native desktop app. Built by .github/workflows/native-bundle.yml with
;   makensis /DVERSION=0.2.3 /DOUTFILE=C:\x\fLLM_0.2.3_x64-setup.exe /DSRC=C:\x\fllm-desktop fllm.nsi
; SRC is the folder of the bundle (fllm-app.exe, fllm-rs.exe, licenses, variant.txt). It installs for the current user
; (no administrator rights; the app updates itself in place and needs a folder it may write), adds a Start menu entry and a
; desktop shortcut, and registers an uninstaller. `/S` installs and uninstalls silently, `/D=<folder>` chooses the folder.
Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"

!ifndef VERSION
  !error "pass /DVERSION=x.y.z"
!endif
!ifndef OUTFILE
  !error "pass /DOUTFILE=<installer file>"
!endif
!ifndef SRC
  !error "pass /DSRC=<bundle folder>"
!endif

!define PRODUCT "fLLM"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\fLLM"

Name "${PRODUCT}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\${PRODUCT}"
InstallDirRegKey HKCU "Software\${PRODUCT}" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
BrandingText "${PRODUCT} ${VERSION}"
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${PRODUCT}"
VIAddVersionKey "FileDescription" "${PRODUCT} setup"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Marco Lutz"

!define MUI_ICON "fllm.ico"
!define MUI_UNICON "fllm.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\fllm-app.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Start ${PRODUCT}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "Install"
  SetOutPath "$INSTDIR"
  ; a running copy keeps its files: say so instead of leaving a half-installed folder
  ClearErrors
  File /r "${SRC}\*.*"
  IfErrors 0 +3
    MessageBox MB_ICONSTOP "Could not write the files. Close ${PRODUCT} if it is running and try again."
    Abort
  File "fllm.ico"
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKCU "Software\${PRODUCT}" "InstallDir" "$INSTDIR"

  CreateShortCut "$SMPROGRAMS\${PRODUCT}.lnk" "$INSTDIR\fllm-app.exe" "" "$INSTDIR\fllm.ico"
  CreateShortCut "$DESKTOP\${PRODUCT}.lnk" "$INSTDIR\fllm-app.exe" "" "$INSTDIR\fllm.ico"

  WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "${PRODUCT}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTKEY}" "Publisher" "Marco Lutz"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\fllm.ico"
  WriteRegStr HKCU "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKCU "${UNINSTKEY}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  ; only a folder that is really ours is emptied: a folder picked by mistake (Documents) must survive
  ${IfNot} ${FileExists} "$INSTDIR\fllm-app.exe"
    MessageBox MB_ICONSTOP "${PRODUCT} is not installed in $INSTDIR."
    Abort
  ${EndIf}
  Delete "$SMPROGRAMS\${PRODUCT}.lnk"
  Delete "$DESKTOP\${PRODUCT}.lnk"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTKEY}"
  DeleteRegKey HKCU "Software\${PRODUCT}"
SectionEnd
