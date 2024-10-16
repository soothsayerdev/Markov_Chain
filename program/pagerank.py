import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000

class Graph:
    def __init__(self):
        # Dictionare to store nodes (pages) and edges (links)
        self.graph = {}
        
    def add_node(self, page):
        """Adiciona uma pagina como nó no grafo"""
        if page not in self.graph:
            self.graph[page] = set()
    
    def add_edge(self, page_from, page_to):
        """Adiciona uma aresta (link) entre duas paginas"""
        if page_from in self.graph:
            self.graph[page_from].add(page_to)
        else:
            self.graph[page_from] = {page_to}
    
    def get_neighbors(self, page):
        """Retorna todas as paginas que a pagina atual referencia (links)"""
        return self.graph.get(page, set())
    
    def pages(self):
        """Retorna todas as paginas (nós) no grafo"""
        return self.graph.keys()
    
    def num_pages(self):
        """Retorna o numero de paginas no grafo"""
        return len(self.graph)
    
    def has_edges(self, page):
        """Retorna True se a pagine tiver links para outras paginas. False se não tiver"""
        return len(self.graph.get(page, [])) > 0
        
def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus  
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages

def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next, given a current page.
    With probability `damping_factor`, choose a link at random linked to by `page`. With probability `1 - damping_factor`, choose a link at random chosen from all pages in the corpus.
    """
    # Probality distribution dict:
    prob_dist = {page_name : 0 for page_name in corpus}
    
    if len(corpus[page]) == 0:
        for page_name in prob_dist:
            prob_dist[page_name] = 1 / len(corpus)
        return prob_dist
        
    # Probability of picking any page at random:
    random_prob = (1 - damping_factor) / len(corpus)
    
    # Probability of picking
    link_prob = damping_factor / len(corpus[page])
    
    # Add probabilities to the distribution:
    for page_name in prob_dist:
        prob_dist[page_name] += random_prob
        
        if page_name in corpus[page]:
            prob_dist[page_name] += link_prob
        
    return prob_dist
        
def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are their estimated PageRank value (a value between 0 and 1). All PageRank values should sum to 1.
    """
    visits = {page_name: 0 for page_name in corpus}
    
    # First page choice is picked at random:
    curr_page = random.choice(list(visits))
    visits[curr_page] += 1
    
    # For remaining n-1 samples, pick the page based on the transistion model:
    for _ in range(0, n - 1):
        trans_model = transition_model(corpus, curr_page, damping_factor)
        
        # Pick next page based on the transition model probabilities:
        rand_val = random.random()
        total_prob = 0
        
        for page_name, probability in trans_model.items():
            total_prob += probability
            
            if rand_val <= total_prob:
                curr_page = page_name
                break
        
        visits[curr_page] += 1
        
    # Normalize visits using sample number:
    page_ranks = {page_name: (visit_num / n) for page_name, visit_num in visits.items()}
    
    print('Sum of sample page ranks: ', round(sum(page_ranks.values()), 4))
    
    return page_ranks    
        
def recursive_pagerank_with_graph(graph, damping_factor, page_ranks = None, iterations = 0, max_rank_change = 1):
    """
    Retorna os valores do PageRank para cada pagina no grafo, atualizando recursivamente até a convergencia.
    Usa o grafo para representar as paginas e seus links
    """
    
    # Base Case: init calculation in first calling
    if page_ranks is None:
        num_pages = graph.num_pages()
        init_rank = 1 / num_pages
        random_choice_prob = (1 - damping_factor) / num_pages
        
        # Init the PageRanks's dict with equals values to all pages
        page_ranks = {page_name: init_rank for page_name in graph.pages()}
        
        return recursive_pagerank_with_graph(graph, damping_factor, page_ranks, iterations, max_rank_change)
    
    # Stop Condition if max change is less then the limit 0.001, return.
    if max_rank_change <= 0.001:
        print(f"Number of iterations: {iterations}")
        print(f"Sum of recursive page ranks: {round(sum(page_ranks.values()), 4)}")
        return page_ranks
    
    # Calculate new values of PageRank
    new_ranks = {page_name: 0 for page_name in graph.pages()}
    max_rank_change = 0
    num_pages = graph.num_pages()
    random_choice_prob = (1 - damping_factor) / num_pages
    
    for page_name in graph.pages():
        surf_choice_prob = 0
        for other_page in graph.pages():
            neighbors = graph.get_neighbors(other_page)
            
            # If other page has no links, it is treated like if it has links to all pages
            if not graph.has_edges(other_page):
                surf_choice_prob += page_ranks[other_page] / num_pages

            # If other page has a link to current page
            elif page_name in neighbors:
                surf_choice_prob += page_ranks[other_page] / len(neighbors)
        
        # New value of PageRank to the current page
        new_rank = random_choice_prob + (damping_factor * surf_choice_prob)
        new_ranks[page_name] = new_rank
    
    # Normalizate new values of PageRank so that they add up 1
    norm_factor = sum(new_ranks.values())
    new_ranks = {page: (rank / norm_factor) for page, rank in new_ranks.items()} 
    
    for page_name in graph.pages():
        rank_change = abs(page_ranks[page_name] - new_ranks[page_name])
        if rank_change > max_rank_change:
            max_rank_change = rank_change
         
    # Return to next recursive iteration
    return recursive_pagerank_with_graph(graph, damping_factor, new_ranks, iterations + 1, max_rank_change)
                
def convert_to_graph(corpus):
    graph = Graph()
    for page in corpus:
        graph.add_node(page)
        for link in corpus[page]:
            graph.add_edge(page, link)
    return graph

if __name__ == "__main__":  

    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])
    
    # Convert to graph
    graph = convert_to_graph(corpus)

    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")    

    print(f"PageRank Results from Iteration")
    ranks = recursive_pagerank_with_graph(graph, DAMPING)
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")   