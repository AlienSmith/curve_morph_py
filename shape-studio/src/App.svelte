<script>
  import { Container, Nav, NavItem, NavLink } from "sveltestrap";
  import Canvas from "./lib/components/Canvas.svelte";
  import Preview from "./lib/components/Preview.svelte";

  let activeTab = "editor"; // 'editor' or 'preview'
</script>

<div class="vh-100 d-flex flex-column bg-black text-white">
  <header
    class="p-3 border-bottom border-secondary bg-black d-flex justify-content-between align-items-center"
  >
    <h5 class="m-0 text-primary fw-bold">SHAPE STUDIO</h5>
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
    <div class="text-secondary small">RTX 5070 Ti Online</div>
  </header>

  <main class="flex-grow-1 overflow-hidden">
    {#if activeTab === "editor"}
      <Canvas />
    {:else}
      <Preview />
    {/if}
  </main>
</div>
