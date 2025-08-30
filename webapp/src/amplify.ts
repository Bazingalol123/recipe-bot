import { Amplify } from 'aws-amplify';

const normalizeHost = (h: string) =>
  (h || '').replace(/^https?:\/\//, '').replace(/\/+$/, '');

const domain = normalizeHost(import.meta.env.VITE_COGNITO_DOMAIN);
const currentOrigin = `${window.location.origin}/`;   // e.g. https://d2s7k5uevi4xe8.cloudfront.net/

const clientId = import.meta.env.VITE_COGNITO_USER_POOL_CLIENT_ID; // SPA client id
const poolId   = import.meta.env.VITE_COGNITO_USER_POOL_ID;

console.log('[Amplify OAuth cfg]', { domain, currentOrigin, clientId, poolId });

Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: poolId,
      userPoolClientId: clientId,
      loginWith: {
        oauth: {
          domain,                           // host *only*
          scopes: ['openid', 'email', 'profile'],
          redirectSignIn:  [currentOrigin], // must be in Cognito “Allowed callback URLs”
          redirectSignOut: [currentOrigin], // must be in “Allowed sign-out URLs”
          responseType: 'code',
        },
      },
    },
  },
});
