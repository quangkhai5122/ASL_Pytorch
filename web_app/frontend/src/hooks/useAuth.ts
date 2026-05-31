import { useState, useCallback } from 'react';

export interface UseAuthReturn {
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  username: string | null;
  token: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  clearError: () => void;
}

// Fixed dev token — matches BYPASS_TOKEN check in backend routes.py
export const DEV_TOKEN = 'dev-bypass-token-local';

/**
 * Auth bypass for local development.
 * isAuthenticated = true by default → no login screen.
 * token = DEV_TOKEN → WebSocket connects without real JWT.
 */
export function useAuth(): UseAuthReturn {
  const [isAuthenticated, setIsAuthenticated] = useState(true);
  const [isLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>('user');

  const login = useCallback(async (user: string, _pass: string) => {
    setIsAuthenticated(true);
    setUsername(user);
  }, []);

  const logout = useCallback(async () => {
    setIsAuthenticated(false);
    setUsername(null);
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return {
    isAuthenticated,
    isLoading,
    error,
    username,
    token: DEV_TOKEN,
    login,
    logout,
    clearError,
  };
}
