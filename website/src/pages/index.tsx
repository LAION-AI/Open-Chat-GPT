import Head from "next/head";
import { CallToAction } from "src/components/CallToAction";
import { Faq } from "src/components/Faq";
import { Footer } from "src/components/Footer";
import { Header } from "src/components/Header";
import { Hero } from "src/components/Hero";

const Home = () => {
  return (
    <>
      <Head>
        <title>Open Assistant</title>
        <meta
          name="description"
          content="Conversational AI for everyone. An open source project to create a chat enabled GPT LLM run by LAION and contributors around the world."
        />
      </Head>
      <main className="oa-basic-theme">
        <Hero />
        <CallToAction />
        <Faq />
      </main>
    </>
  );
};

Home.getLayout = (page) => (
  <div className="grid grid-rows-[min-content_1fr_min-content] h-full justify-items-stretch">
    <Header transparent={true} />
    {page}
    <Footer />
  </div>
);

export default Home;
