import rewireMobx from 'react-app-reqire-mobx';
const { compose } = require('react-app-rewired');

module.exports = function(config, env){
  const rewires = compose(
    rewireMobx
  );
  // do custom config
  // ...
  return rewires(config, env);
}
