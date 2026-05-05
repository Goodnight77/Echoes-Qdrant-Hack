import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Route, Switch } from 'wouter';
import Layout from './components/Layout';
import SearchPage from './pages/SearchPage';
import PeoplePage from './pages/PeoplePage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1 },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Layout>
        <Switch>
          <Route path="/" component={SearchPage} />
          <Route path="/search" component={SearchPage} />
          <Route path="/people" component={PeoplePage} />
        </Switch>
      </Layout>
    </QueryClientProvider>
  );
}
