# Paper2Product 2.0 Deployment Guide

This directory contains production-ready deployment configurations for Paper2Product 2.0, with separate setups for the Next.js frontend and Python backend.

## Overview

- **Frontend**: Deployed on Vercel (Next.js) - `vercel.json`
- **Backend**: Multi-platform support - Railway (`railway.toml`) or Fly.io (`fly.toml`)

---

## Frontend Deployment (Vercel)

### Configuration: `vercel.json`

The Vercel configuration handles the Next.js frontend with API rewrites to the backend.

**Key Features:**
- Build command runs from the frontend directory
- API rewrites `/api/*` requests to the backend service
- Sets `NEXT_PUBLIC_API_URL` environment variable for frontend API calls
- Cache-Control headers prevent caching of API responses

### Deployment Steps

1. **Connect Repository**
   ```bash
   vercel link
   ```

2. **Set Environment Variables**
   ```bash
   vercel env add NEXT_PUBLIC_API_URL https://your-backend-url.com
   vercel env add BACKEND_URL https://your-backend-url.com
   ```

3. **Deploy**
   ```bash
   vercel deploy --prod
   ```

Or use GitHub integration for automatic deployments on push.

---

## Backend Deployment

### Option 1: Railway

#### Configuration: `railway.toml`

Railway deployment configuration for the Python backend.

**Key Features:**
- Start command: `python -m paper2product`
- Runs on port 8000
- Health checks enabled
- Automatic restart on failure
- `PYTHONUNBUFFERED=1` ensures real-time logs

#### Deployment Steps

1. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   ```

2. **Login**
   ```bash
   railway login
   ```

3. **Initialize Project**
   ```bash
   railway init
   ```

4. **Set Environment Variables**
   ```bash
   railway variable set ENVIRONMENT=production
   railway variable set SECRET_KEY=your-secret-key
   railway variable set DATABASE_URL=your-database-url
   ```

5. **Deploy**
   ```bash
   railway up
   ```

Or push to your linked GitHub repository for automatic deployment.

---

### Option 2: Fly.io

#### Configuration: `fly.toml`

Fly.io deployment configuration optimized for Python backends.

**Key Features:**
- App name: `p2p-backend`
- Internal port: 8000
- Force HTTPS enabled
- Auto-stop enabled for cost optimization
- Health checks at `/health` endpoint
- Connection-based concurrency limits
- 512MB memory, 1 CPU VM

#### Deployment Steps

1. **Install Fly CLI**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**
   ```bash
   fly auth login
   ```

3. **Initialize App** (or use existing app)
   ```bash
   fly launch
   ```

4. **Set Environment Variables**
   ```bash
   fly secrets set ENVIRONMENT=production
   fly secrets set SECRET_KEY=your-secret-key
   fly secrets set DATABASE_URL=your-database-url
   ```

5. **Deploy**
   ```bash
   fly deploy
   ```

---

## Environment Variables

### Frontend (.env.production)
```
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

### Backend (Production)
```
ENVIRONMENT=production
PORT=8000
PYTHONUNBUFFERED=1
SECRET_KEY=your-secret-key-here
DATABASE_URL=your-database-url
```

---

## Health Checks

### Fly.io
- Expects a `/health` endpoint returning 200 OK
- Interval: 10 seconds
- Timeout: 5 seconds

### Railway
- Health checks enabled by default
- Configure by adding health check endpoints in `railway.toml`

---

## Monitoring & Logs

### Vercel
```bash
vercel logs --follow
```

### Railway
```bash
railway logs --follow
```

### Fly.io
```bash
fly logs --follow
```

---

## Cost Optimization

### Vercel
- Generous free tier with automatic scaling
- Pay as you grow model

### Railway
- $5/month starter credit
- Pay for compute and data transfer
- Auto-sleep feature available

### Fly.io
- Free tier with up to 3 shared CPUs
- `auto_stop = true` in `fly.toml` stops machines when not in use
- Saves on costs during off-peak hours

---

## Custom Domain Setup

### All Platforms
1. Purchase domain or use existing
2. Update DNS records to point to your deployment
3. Configure SSL/TLS (automatic on all platforms)
4. Update `NEXT_PUBLIC_API_URL` on Vercel to use your domain

---

## Troubleshooting

### Frontend fails to build
- Check `frontend/package.json` for build script
- Verify all environment variables are set
- Review build logs in Vercel dashboard

### Backend won't start
- Ensure `python -m paper2product` is valid entry point
- Check `/health` endpoint exists for health checks
- Verify port 8000 is not bound locally

### API calls return 502/503
- Check backend logs for errors
- Verify backend URL in `NEXT_PUBLIC_API_URL`
- Ensure CORS is properly configured on backend

---

## Documentation Links

- [Vercel Deployment Docs](https://vercel.com/docs)
- [Railway Deployment Docs](https://docs.railway.app/)
- [Fly.io Deployment Docs](https://fly.io/docs/)
