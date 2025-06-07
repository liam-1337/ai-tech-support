export interface IElectronAPI {
  openDirectory: () => Promise<string | undefined>; // Adjusted to return string | undefined as per main.js logic
}

declare global {
  interface Window {
    electronAPI: IElectronAPI;
  }
}
