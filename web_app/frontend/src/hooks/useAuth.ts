import { useState, useCallback } from 'react';

export interface UseAuthReturn {
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  username: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  clearError: () => void;
}

/**
 * Auth bypass for local development.
 * isAuthenticated = true by default → no login screen.
 */
export function useAuth(): UseAuthReturn {
  const [isAuthenticated, setIsAuthenticated] = useState(true);
  const [isLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>("user");

  const login = useCallback(async (user: string, _pass: string) => {
    setIsAuthenticated(true);
    setUsername(user);
  }, []);

  const logout = useCallback(async () => {
    setIsAuthenticated(false);
    setUsername(null);
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return { isAuthenticated, isLoading, error, username, login, logout, clearError };
}
