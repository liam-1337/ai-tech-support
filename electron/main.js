const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const isDev = require('electron-is-dev')

let pythonServerProcess
let nextServerProcess

// --- Application Menu ---
const menuTemplate = [
  {
    label: 'File',
    submenu: [
      {
        label: 'Open Directory...',
        click: async () => {
          const focusedWindow = BrowserWindow.getFocusedWindow()
          if (focusedWindow) {
            const { canceled, filePaths } = await dialog.showOpenDialog(focusedWindow, {
              properties: ['openDirectory']
            })
            if (!canceled && filePaths.length > 0) {
              console.log('Directory selected via menu:', filePaths[0])
              // Optionally send to renderer: focusedWindow.webContents.send('directory-selected', filePaths[0]);
            }
          } else {
            // Fallback if no window is focused (e.g. before first window opens)
             const { canceled, filePaths } = await dialog.showOpenDialog({
              properties: ['openDirectory']
            })
            if (!canceled && filePaths.length > 0) {
              console.log('Directory selected via menu (no focused window):', filePaths[0])
            }
          }
        }
      },
      { type: 'separator' },
      { role: 'quit' }
    ]
  },
  {
    label: 'Edit',
    submenu: [
      { role: 'undo' },
      { role: 'redo' },
      { type: 'separator' },
      { role: 'cut' },
      { role: 'copy' },
      { role: 'paste' },
      { role: 'selectAll' }
    ]
  },
  {
    label: 'View',
    submenu: [
      { role: 'reload' },
      { role: 'forceReload' },
      { role: 'toggleDevTools' },
      { type: 'separator' },
      { role: 'resetZoom' },
      { role: 'zoomIn' },
      { role: 'zoomOut' },
      { type: 'separator' },
      { role: 'togglefullscreen' }
    ]
  },
  {
    label: 'Help',
    submenu: [
      {
        label: 'About',
        click: async () => {
          // const { shell } = require('electron')
          // await shell.openExternal('https://electronjs.org')
          dialog.showMessageBox(null, {
            type: 'info',
            title: 'About AI Tech Support Agent',
            message: 'AI Tech Support Agent',
            detail: 'Version 1.0.0\nAn AI-powered technical support assistant.'
          });
        }
      }
    ]
  }
];

if (process.platform === 'darwin') {
  menuTemplate.unshift({
    label: app.name,
    submenu: [
      { role: 'about' },
      { type: 'separator' },
      { role: 'services' },
      { type: 'separator' },
      { role: 'hide' },
      { role: 'hideOthers' },
      { role: 'unhide' },
      { type: 'separator' },
      { role: 'quit' }
    ]
  });

  // Edit menu (macOS)
  const editMenu = menuTemplate.find(m => m.label === 'Edit');
  if (editMenu) {
    editMenu.submenu.push(
      { type: 'separator' },
      { label: 'Speech', submenu: [{ role: 'startSpeaking' }, { role: 'stopSpeaking' }] }
    );
  }

  // Window menu (macOS) - Basic
   menuTemplate.splice(3, 0, { // Insert after View
    label: 'Window',
    submenu: [
      { role: 'close' },
      { role: 'minimize' },
      { role: 'zoom' },
      { type: 'separator' },
      { role: 'front' }
    ]
  });
}

const menu = Menu.buildFromTemplate(menuTemplate)
Menu.setApplicationMenu(menu)
// --- End Application Menu ---


function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js') // Ensure preload path is correct
    }
  })

  win.loadURL('http://localhost:3000') // Load the Next.js frontend
}

// IPC handler for opening directory dialog
ipcMain.handle('open-directory-dialog', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openDirectory']
  })
  if (canceled || filePaths.length === 0) {
    return undefined
  } else {
    return filePaths[0] // Return the first selected path
  }
})

app.whenReady().then(async () => {
  let pythonScriptPath
  if (isDev) {
    // In development, __dirname is electron/
    pythonScriptPath = path.join(__dirname, '..', 'ai_tech_support_agent', 'app', 'main.py')
  } else {
    // In production, app.getAppPath() is the root of the unpacked ASAR or app directory
    // Assuming 'ai_tech_support_agent' is copied to the root of the app contents by 'files'
    pythonScriptPath = path.join(app.getAppPath(), 'ai_tech_support_agent', 'app', 'main.py')
  }

  console.log(`Attempting to start Python server from: ${pythonScriptPath}`)
  pythonServerProcess = spawn('python', [pythonScriptPath])

  pythonServerProcess.stdout.on('data', (data) => {
    console.log(`Python Server stdout: ${data}`)
  })

  pythonServerProcess.stderr.on('data', (data) => {
    console.error(`Python Server stderr: ${data}`)
  })

  pythonServerProcess.on('close', (code) => {
    console.log(`Python Server exited with code ${code}`)
  })

  if (!isDev) {
    // Start Next.js server in production
    // Assumes 'frontend/.next/standalone' is copied to the root of app contents by 'files'
    const nextServerPath = path.join(app.getAppPath(), 'frontend', '.next', 'standalone', 'server.js')
    const nextServerWorkingDirectory = path.join(app.getAppPath(), 'frontend', '.next', 'standalone')

    console.log(`Attempting to start Next.js server from: ${nextServerPath} with CWD: ${nextServerWorkingDirectory}`)
    // Critical: Set CWD for server.js, it expects to be run from its own directory
    nextServerProcess = spawn('node', [nextServerPath], { cwd: nextServerWorkingDirectory })

    nextServerProcess.stdout.on('data', (data) => {
      console.log(`Next.js Server stdout: ${data}`)
      // Potentially wait for a specific message indicating server readiness before createWindow if needed
    })
    nextServerProcess.stderr.on('data', (data) => {
      console.error(`Next.js Server stderr: ${data}`)
    })
    nextServerProcess.on('close', (code) => {
      console.log(`Next.js Server exited with code ${code}`)
    })
    // Add a small delay or use wait-on for http://localhost:3000 before createWindow
    // For simplicity, adding a small timeout. Robust solution would use wait-on or check stdout.
    await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5s for Next.js to start
  }

  createWindow()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('will-quit', () => {
  if (pythonServerProcess) {
    console.log('Attempting to kill Python server process...')
    pythonServerProcess.kill()
  }
  if (nextServerProcess) {
    console.log('Attempting to kill Next.js server process...')
    nextServerProcess.kill()
  }
})
