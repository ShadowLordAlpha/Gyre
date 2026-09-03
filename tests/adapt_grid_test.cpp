#include "gyre/adapt/agent.hpp"

#include <gtest/gtest.h>

TEST(Adapt, GridworldDeterministic) {
  gyre::adapt::GridWorld env;
  gyre::Rng rng(1);
  auto o = env.reset(rng);
  ASSERT_TRUE(o) << (o ? "" : o.error().message);
  gyre::adapt::Action a{gyre::adapt::ActionSpace::discrete_int, 2, {}};
  auto s = env.step(a);
  ASSERT_TRUE(s);
  auto p = s->observation.host_span<float>();
  ASSERT_TRUE(p);
  EXPECT_EQ((*p)[0], 1.f);
}

TEST(Adapt, PolicyAgentAct) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(4);
  auto ag = gyre::adapt::PolicyAgent::create(2, 4, *d, rng);
  ASSERT_TRUE(ag);
  gyre::adapt::GridWorld env;
  auto o = env.reset(rng);
  ASSERT_TRUE(o);
  auto act = ag->act(*o, rng);
  ASSERT_TRUE(act) << (act ? "" : act.error().message);
  EXPECT_EQ(act->space, gyre::adapt::ActionSpace::discrete_int);
}
