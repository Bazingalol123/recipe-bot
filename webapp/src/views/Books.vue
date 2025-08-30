<template>
  <main style="padding:2rem">
    <h2>My Books</h2>
    <p v-if="loading">Loading…</p>
    <p v-if="error" style="color:#b00">{{ error }}</p>
    <ul v-if="!loading && !error">
      <li v-for="b in books" :key="b.book_id">{{ b.title }}</li>
    </ul>
  </main>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { fetchAuthSession } from 'aws-amplify/auth';

type Book = { book_id: string; title: string };

const books = ref<Book[]>([]);
const loading = ref(true);
const error = ref<string | null>(null);

onMounted(async () => {
  const base = import.meta.env.VITE_API_BASE_URL;
  try {
    const idToken = (await fetchAuthSession()).tokens?.idToken?.toString();
    if (!idToken) throw new Error('No ID token (are you signed in?)');

   const res = await fetch(`${base}/books`, {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${idToken}`,
        "Content-Type": "application/json",
      },
      mode: "cors",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    books.value = await res.json();
  } catch (e: any) {
    error.value = e?.message ?? String(e);
  } finally {
    loading.value = false;
  }
});
</script>
