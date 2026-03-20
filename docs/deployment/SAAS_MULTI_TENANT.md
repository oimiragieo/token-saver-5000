# Multi-Tenant SaaS Deployment

Use this guide when Token Saver 5000 is running as shared infrastructure for multiple teams, customers, or internal agents.

## Scope Model

The product uses four scope fields to enforce isolation:

1. `workspace_id`: top-level tenant or workspace boundary.
2. `user_id`: end-user boundary inside a workspace.
3. `agent_id`: automation or assistant boundary inside a workspace.
4. `session_id`: short-lived conversation or workflow boundary.

These fields appear across the MCP tool surface and should be forwarded consistently by any web tier, HTTP server, or job worker that fronts Token Saver 5000.

## Isolation Rules

1. Always treat `workspace_id` as the primary isolation key for multi-tenant traffic.
2. Use `user_id` to separate user-specific memories, prompt state, and personalization.
3. Use `agent_id` when multiple assistants share one workspace but should not leak state.
4. Use `session_id` for short-lived conversational state, imports, exports, and replayable workflows.

If one of these fields is omitted, the request becomes broader. In a SaaS deployment, that should be a conscious platform decision rather than an accidental default.

## Recommended Request Routing

For HTTP or queue-based deployments:

1. Authenticate the caller at the API gateway or reverse proxy.
2. Resolve the caller to a `workspace_id`.
3. Attach `user_id` when the request is user-owned.
4. Attach `agent_id` for automated workers or role-based assistants.
5. Attach `session_id` when the operation belongs to a bounded conversation or job run.

This keeps downstream tools like memory, prompt registry, connector feeds, temporal exports, and handoff bundles aligned with the same isolation model.

## Suggested Service Patterns

### Single-tenant internal deployment

- One trusted team.
- `workspace_id` can be fixed or omitted intentionally.
- Good fit for stdio MCP plus local persistence.

### Multi-tenant web/API deployment

- Many customers behind one HTTP server or worker pool.
- `workspace_id` must be injected on every request.
- Use `user_id` and `agent_id` for finer isolation.
- Put auth, rate limiting, and audit logging in front of the service.

### Agent platform deployment

- One workspace may have many agents.
- Use `agent_id` to isolate memory and prompt state per agent role.
- Use `session_id` to separate parallel runs and replay bundles.

## Operational Checklist

1. Validate scope fields at the edge before calling internal handlers.
2. Log resolved `workspace_id`, `user_id`, `agent_id`, and `session_id` in structured request metadata.
3. Keep customer-specific persistence paths and backups separated.
4. Test cross-workspace isolation regularly with launch-readiness suites.
5. Review `docs/guides/WORKFLOW_ORCHESTRATION.md` so upstream callers preserve the right sequencing.

## Related Docs

1. `README.md`
2. `docs/deployment/DOCKER.md`
3. `docs/deployment/DEPLOYMENT.md`
4. `docs/guides/WORKFLOW_ORCHESTRATION.md`
