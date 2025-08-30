<template>
  <main style="padding:2rem">
    <h1>Sign in</h1>
    <p>Use your Google account.</p>

    <!-- Primary: Amplify redirect -->
    <button @click="google">Continue with Google</button>

    <!-- Fallback: direct Hosted UI link (shows only if we built it) -->
    <p v-if="authorizeUrl" style="margin-top:12px;opacity:.7">
      Trouble? <a :href="authorizeUrl">Open Hosted UI directly</a>
    </p>
  </main>
</template>

<script setup lang="ts">
import { signInWithRedirect, fetchAuthSession } from 'aws-amplify/auth';
import { onMounted, computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';

// Build a backup authorize URL from the same values Amplify will use
const domain   = (import.meta.env.VITE_COGNITO_DOMAIN || '').replace(/^https?:\/\//, '').replace(/\/+$/, '');
const clientId = import.meta.env.VITE_COGNITO_USER_POOL_CLIENT_ID;
const redirect = `${window.location.origin}/`;  // same-origin guarantee

const authorizeUrl = computed(() =>
  (domain && clientId)
    ? `https://${domain}/oauth2/authorize?client_id=${encodeURIComponent(clientId)}&response_type=code&redirect_uri=${encodeURIComponent(redirect)}&scope=openid+email+profile&identity_provider=Google`
    : ''
);

const route = useRoute();
const router = useRouter();

onMounted(async () => {
  try {
    const s = await fetchAuthSession();
    if (s?.tokens?.idToken) {
      const next = (route.query.next as string) || '/';
      router.replace(next);
    }
  } catch (e) {
    console.warn('[fetchAuthSession] no session yet', e);
  }
});

const google = async () => {
  try {
    await signInWithRedirect({ provider: 'Google' });
  } catch (e) {
    console.error('[signInWithRedirect] failed', e);
    // as a last resort, navigate to Hosted UI directly:
    if (authorizeUrl.value) window.location.href = authorizeUrl.value;
  }
};
</script>
