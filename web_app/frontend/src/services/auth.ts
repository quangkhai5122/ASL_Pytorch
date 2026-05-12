import { apiClient, LoginResponse, TokenData } from './api';

export class AuthService {
  /**
   * Login with username and password
   */
  async login(username: string, password: string): Promise<LoginResponse> {
    try {
      const response = await apiClient.login(username, password);
      return response;
    } catch (error) {
      console.error('Login failed:', error);
      throw new Error('Login failed. Please check your credentials.');
    }
  }

  /**
   * Verify current token is valid
   */
  async verifyToken(): Promise<TokenData> {
    try {
      return await apiClient.verifyToken();
    } catch (error) {
      console.error('Token verification failed:', error);
      throw new Error('Session expired. Please login again.');
    }
  }

  /**
   * Logout current user
   */
  async logout(): Promise<void> {
    await apiClient.logout();
  }

  /**
   * Check if user is currently authenticated
   */
  isAuthenticated(): boolean {
    return apiClient.isAuthenticated();
  }

  /**
   * Get stored JWT token
   */
  getToken(): string | null {
    return apiClient.getToken();
  }

  /**
   * Set JWT token (useful when token is provided externally)
   */
  setToken(token: string): void {
    apiClient.setToken(token);
  }

  /**
   * Get test credentials (development only)
   */
  async getTestCredentials(): Promise<Array<{ username: string; password: string }>> {
    try {
      const response = await apiClient.getTestCredentials();
      return response.credentials;
    } catch (error) {
      console.error('Failed to get test credentials:', error);
      return [];
    }
  }

  /**
   * Auto-login with test credentials (development)
   */
  async autoLoginDev(credentialIndex: number = 0): Promise<LoginResponse | null> {
    try {
      const credentials = await this.getTestCredentials();
      if (credentials.length > credentialIndex) {
        const { username, password } = credentials[credentialIndex];
        return await this.login(username, password);
      }
    } catch (error) {
      console.error('Auto-login failed:', error);
    }
    return null;
  }
}

export const authService = new AuthService();
