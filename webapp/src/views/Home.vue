<script setup lang="ts">
import { ref } from 'vue'

const url = ref('')
const recipe = ref<any|null>(null)
const loading = ref(false)
const error = ref<string|null>(null)

const fetchRecipe = async () => {
  loading.value = true
  error.value = null
  recipe.value = null
  try {
    const resp = await fetch(import.meta.env.VITE_BOT_RECIPE_LAMBDA_URL,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.value })
      }
    )
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    recipe.value = await resp.json()
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <main>
    <h2>Recipe Generator</h2>
    <input v-model="url" placeholder="Paste Instagram/TikTok/Facebook link" />
    <button @click="fetchRecipe" :disabled="loading">Generate</button>

    <p v-if="loading">⏳ Processing…</p>
    <p v-if="error" style="color:red">{{ error }}</p>

    <div v-if="recipe">
      <h3>{{ recipe.title }}</h3>
      <p>Servings: {{ recipe.servings }}</p>
      <ul>
        <li v-for="ing in recipe.ingredients" :key="ing.item">
          {{ ing.item }} — {{ ing.quantity }} {{ ing.unit }}
        </li>
      </ul>
      <ol>
        <li v-for="s in recipe.steps" :key="s.number">{{ s.instruction }}</li>
      </ol>
    </div>
  </main>
</template>
