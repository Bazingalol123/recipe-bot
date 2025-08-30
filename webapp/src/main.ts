import { createApp } from 'vue';
import App from './App.vue';
import router from './router.ts';
import './amplify';
import '@aws-amplify/ui-vue/styles.css';

createApp(App).use(router).mount('#app');
