FROM nginx:alpine

# Copy static site
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80

# Run nginx in foreground
CMD ["nginx", "-g", "daemon off;"]