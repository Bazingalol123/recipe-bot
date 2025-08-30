<template>
  <nav style="padding:12px; display:flex; gap:16px; border-bottom:1px solid #eee">
    <router-link to="/">Home</router-link>
    <router-link to="/books">My Books</router-link>
    <span style="margin-left:auto"></span>
    <span v-if="email" style="opacity:.8">{{ email }}</span>
    <button v-if="!email" @click="login" style="margin-left:8px">Sign in</button>
    <button v-else @click="logout" style="margin-left:8px">Sign out</button>
  </nav>
  <router-view />
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { fetchAuthSession, signInWithRedirect, signOut } from 'aws-amplify/auth';

const email = ref<string | null>(null);

onMounted(async () => {
  // Resolve redirect & load tokens; v6 does this via fetchAuthSession
  try {
    const s = await fetchAuthSession();
    const id = s?.tokens?.idToken?.toString();
    if (id) {
      const payload = JSON.parse(atob(id.split('.')[1]));
      email.value = payload?.email ?? null;
    }
  } catch {}
});

const login = () => signInWithRedirect({ provider: 'Google' });
const logout = () => signOut();
</script>

<style>
html, body, #app { height: 100%; margin: 0; font-family: system-ui, sans-serif; }
a { text-decoration: none; }
</style>
