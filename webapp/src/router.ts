import { createRouter, createWebHistory } from 'vue-router';
import Home from './views/Home.vue';
import Login from './views/Login.vue';
import Books from './views/Books.vue';
import { fetchAuthSession } from 'aws-amplify/auth';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: Home },
    { path: '/login', component: Login },
    { path: '/books', component: Books, meta: { requiresAuth: true } },
  ],
});

router.beforeEach(async (to) => {
  if (!to.meta.requiresAuth) return true;
  try {
    const s = await fetchAuthSession();
    if (s?.tokens?.idToken) return true;
  } catch {}
  return { path: '/login', query: { next: to.fullPath } };
});

export default router;
