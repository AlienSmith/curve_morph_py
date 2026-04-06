<script>
  import { Container, Nav, NavItem, NavLink } from "sveltestrap";
  // Updated Import Paths
  import Editor from "./lib/components/Editor.svelte";
  import Preview from "./lib/components/Preview.svelte";

  let activeTab = "editor"; // 'editor' or 'preview'
</script>

<div class="vh-100 d-flex flex-column bg-black text-white">
  <header
    class="p-3 border-bottom border-secondary bg-black d-flex justify-content-between align-items-center"
  >
    <div class="d-flex align-items-center gap-3">
      <h5 class="m-0 text-primary fw-bold">SHAPE STUDIO</h5>
      <span
        class="badge border border-primary text-primary small"
        style="font-size: 0.7rem;">V2.0</span
      >
    </div>

    <Nav pills>
      <NavItem>
        <NavLink
          active={activeTab === "editor"}
          on:click={() => (activeTab = "editor")}
          style="cursor: pointer;"
        >
          Editor
        </NavLink>
      </NavItem>
      <NavItem>
        <NavLink
          active={activeTab === "preview"}
          on:click={() => (activeTab = "preview")}
          style="cursor: pointer;"
        >
          Previewer
        </NavLink>
      </NavItem>
    </Nav>

    <div class="d-flex align-items-center gap-2">
      <div class="status-dot"></div>
      <div class="text-secondary small">RTX 5070 Ti Online</div>
    </div>
  </header>

  <main class="flex-grow-1 overflow-hidden">
    {#if activeTab === "editor"}
      <Editor />
    {:else}
      <Preview />
    {/if}
  </main>
</div>

<style>
  /* Status pulse for that "Studio" feel */
  .status-dot {
    width: 8px;
    height: 8px;
    background-color: #00ffcc;
    border-radius: 50%;
    box-shadow: 0 0 8px #00ffcc;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0% {
      opacity: 1;
    }
    50% {
      opacity: 0.4;
    }
    100% {
      opacity: 1;
    }
  }

  /* Ensure the main area takes up all remaining space */
  main {
    display: flex;
    flex-direction: column;
  }
</style>
