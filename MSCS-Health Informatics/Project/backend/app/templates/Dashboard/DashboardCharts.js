var hts_test_pos_bar = "HTS_TEST_POS-BAR.json";
var tx_new_bar = "TX_NEW-BAR.json";
var net_cum_bar = "NET_CUM-BAR.json";
var progress_goal_bar = "PROGRESS_GOAL-BAR.json";
var data_table = './DATA_TABLE.json';
var case_finding_bar = "CASE_FINDING-BAR.json";
var linkage_proxy_bar ="LINKAGE_PROXY-BAR.json";
var retention_proxy_bar ="RETENTION_PROXY-BAR.json";
var hts_tst_pos_line = "HTS_TST_POS-LINE.json";
var tx_new_line = "TX_NEW-LINE.json";
var tx_net_new_line = "TX_NET_NEW-LINE.json";

$(function () {
  var ctx = document.getElementById("HTS_TEST_POS-BAR").getContext('2d');
  var hBarOptions = {
      responsive: false,
      maintainAspectRatio: false,
      legend: { display:false },
      scales: {
          xAxes: [{
            stacked: false,
            ticks:{
              beginAtZero:true,
              min: 0,
              max: this.max,// Your absolute max value
              callback: function (value) {
                return (value) + '%'; // convert it to percentage
              },
            }
          }],
          yAxes: [{
            stacked: true,
            ticks:{
              beginAtZero:true
            }
          }]
      },
      title: {
        display: true,
        text: 'HOSPITAL_ADMITTANCE'
      },
      plugins: {
        datalabels: {
          color: '#154360',
          anchor: 'end',
          align: 'top',
          formatter: function (value) {
            return Math.round(value*max/100);
          },
          font: {
            weight: 'bold',
            size: 10
          }
        }
      }
  };
  var hBarData = {
      datasets: [
        {
          data: [],
          backgroundColor: '#E67E22',
        },
        {
          data: [],
          backgroundColor: '#FAD7A0',
        }
      ]
  };
  var chart = new Chart(ctx, {
      type: 'horizontalBar',
      data: hBarData,
      options: hBarOptions
  });
  ajax_overlayBar_chart(chart, hts_test_pos_bar);
});
$(function () {
  var ctx = document.getElementById("TX_NEW-BAR").getContext('2d');
  var hBarOptions = {
      responsive: false,
      maintainAspectRatio: false,
      legend: { display:false },
      scales: {
          xAxes: [{
            stacked: false,
            ticks:{
              beginAtZero:true,
              min: 0,
              max: this.max,// Your absolute max value
              callback: function (value) {
                return (value.toFixed(0)) + '%'; // convert it to percentage
              },
            }
          }],
          yAxes: [{
            stacked: true,
            ticks:{
              beginAtZero:true
            }
          }]
      },
      title: {
        display: true,
        text: 'NET_NEW Cum.'
      },
      plugins: {
        datalabels: {
          color: '#154360',
          anchor: 'end',
          align: 'top',
          formatter: function (value) {
            return Math.round(value*max/100);
          },
          font: {
            weight: 'bold',
            size: 10
          }
        }
      }
  };
  var hBarData = {
      datasets: [
        {
          data: [],
          backgroundColor: '#0067a3',
        },
        {
          data: [],
          backgroundColor: '#4697c7',
        }
      ]
  };
  var chart = new Chart(ctx, {
      type: 'horizontalBar',
      data: hBarData,
      options: hBarOptions
  });
  ajax_overlayBar_chart(chart, tx_new_bar);
});
$(function () {
  var ctx = document.getElementById("NET_CUM-BAR").getContext('2d');
  var hBarOptions = {
      responsive: false,
      maintainAspectRatio: false,
      legend: { display:false },
      scales: {
          xAxes: [{
            stacked: false,
            ticks:{
              beginAtZero:true,
              min: 0,
              max: this.max,// Your absolute max value
              callback: function (value) {
                return (value) + '%'; // convert it to percentage
              },
            }
          }],
          yAxes: [{
            stacked: true,
            ticks:{
              beginAtZero:true
            }
          }]
      },
      title: {
        display: true,
        text: 'TX_NEW Cum.'
      },
      plugins: {
        datalabels: {
          color: '#154360',
          anchor: 'end',
          align: 'top',
          formatter: function (value) {
            return Math.round(value*max/100);
          },
          font: {
            weight: 'bold',
            size: 10
          }
        }
      }
  };
  var hBarData = {
      datasets: [
        {
          data: [],
          backgroundColor: '#E67E22',
        },
        {
          data: [],
          backgroundColor: '#FAD7A0',
        }
      ]
  };
  var chart = new Chart(ctx, {
      type: 'horizontalBar',
      data: hBarData,
      options: hBarOptions
  });
  ajax_overlayBar_chart(chart, net_cum_bar);
});

$(function () {
  var ctx = document.getElementById("PROGRESS_GOAL-BAR").getContext('2d');
  var progressData = {
      labels: [],
      datasets: [{
            data: [],
            backgroundColor: '#f56954'
        },
        {
            data: [],
            backgroundColor: '#3c8dbc'
        }]
  };
  var progressOptions = {
    responsive: false,
    maintainAspectRatio: false,
    legend: {
        position: 'bottom',
        labels: {
            boxWidth: 12
        }
    },
    title:{
      display: true,
      text: 'Progress Towards FY20 Treatment Goal',
      color: '#FAD7A0'
    },
    scales:{
        xAxes: [{
          stacked: true,
          ticks:{
            beginAtZero:false
          }
        }],
        yAxes: [{
          stacked: false,
          ticks:{
            beginAtZero:true,
            min: 0,
            max: this.max,// Your absolute max value
            callback: function (value) {
              return (value) + '%'; // convert it to percentage
            },
          }
        }]
    }
  };
  var chart = new Chart(ctx, {
      type: 'bar',
      data: progressData,
      options: progressOptions
  });
  ajax_overlay_chart(chart, progress_goal_bar);
});
$('#DataTable').DataTable({
    ajax: {
        url: data_table,
        dataSrc: "chartData"
    },
    columns: [
        { data: 'FY19Q2' },
        { data: 'Current' },
        { data: 'Result' },
        { data: 'Targ' },
        { data: 'Yiel' },
        { data: 'Result' },
        { data: 'Targ' },
        { data: 'Yiel' },
        { data: 'Result' },
        { data: 'Targ' },
        { data: 'Link' },
        { data: 'Loss/IG' },
        { data: 'Ret' }
    ],
    pageLength: 10,
    paging: false,
    ordering: false,
    info: false,
    searching: false,
    jQueryUI:true
});

$(function () {
  var ctx = document.getElementById("CASE_FINDING-BAR").getContext('2d');
  var barOptions = {
      responsive: false,
      maintainAspectRatio: false,
      title:{
        display: true,
        text: 'Case Finding',
        color: '#FAD7A0'
      },
      legend: {
          display: false,
          position: 'bottom',
          labels: {
              boxWidth: 12
          }
      },
      scales: {
        yAxes: [{
          id: 'A',
          type: 'linear',
          position: 'left',
          ticks: {
            max: 1000,
            min: 0
          }
        }, {
          id: 'B',
          type: 'linear',
          position: 'right',
          ticks: {
            max: 10,
            min: 0
          }
        }]
      }
  };
  var barData = {
      labels: [],
      datasets: [{
        label: 'A',
        yAxisID:'A',
        data: [],
        backgroundColor: 'blue'
      },{
        label:'B',
        yAxisID:'B',
        data: [],
        backgroundColor:'orange'
      }]
  };
  var chart = new Chart(ctx, {
      type: 'bar',
      data: barData,
      options: barOptions
  });
  ajax_overlay_chart(chart, case_finding_bar);
});
$(function () {
  var ctx = document.getElementById("LINKAGE_PROXY-BAR").getContext('2d');
  var barOptions = {
      responsive: false,
      maintainAspectRatio: false,
      title:{
        display: true,
        text: 'Linkage Proxy',
        color: '#FAD7A0'
      },
      legend: {
          display: false,
          position: 'bottom',
          labels: {
              boxWidth: 12
          }
      },
      scales: {
        yAxes: [{
          id: 'A',
          type: 'linear',
          position: 'left',
          ticks: {
            max: 1000,
            min: 0
          }
        }, {
          id: 'B',
          type: 'linear',
          position: 'right',
          ticks: {
            max: 10,
            min: 0
          }
        }]
      }
  };
  var barData = {
      labels: [],
      datasets: [{
        label: 'A',
        yAxisID:'A',
        data: [],
        backgroundColor: 'blue',
      },{
        label:'B',
        yAxisID:'B',
        data: [],
        backgroundColor:'orange'
      }]
  };
  var chart = new Chart(ctx, {
      type: 'bar',
      data: barData,
      options: barOptions
  });
  ajax_overlay_chart(chart, linkage_proxy_bar);
});
$(function () {
  var ctx = document.getElementById("RETENTION_PROXY-BAR").getContext('2d');
  var barOptions = {
      responsive: false,
      maintainAspectRatio: false,
      title:{
        display: true,
        text: 'Retention Proxy',
        color: '#FAD7A0'
      },
      legend: {
          display: false,
          position: 'bottom',
          labels: {
              boxWidth: 12
          }
      },
      scales: {
        yAxes: [{
          id: 'A',
          type: 'linear',
          position: 'left',
          ticks: {
            max: 1000,
            min: 0
          }
        }, {
          id: 'B',
          type: 'linear',
          position: 'right',
          ticks: {
            max: 10,
            min: 0
          }
        }]
      }
  };
  var barData = {
      labels: [],
      datasets: [{
        label: 'A',
        yAxisID:'A',
        data: [],
        backgroundColor: 'blue',
      },{
        label:'B',
        yAxisID:'B',
        data: [],
        backgroundColor:'orange'
      }]
  };
  var chart = new Chart(ctx, {
      type: 'bar',
      data: barData,
      options: barOptions
  });
  ajax_overlay_chart(chart, retention_proxy_bar);
});

$(function () {
  var ctx = document.getElementById("HTS_TST_POS-LINE").getContext('2d');
  var lineOptions = {
      responsive: false,
      maintainAspectRatio: false,
      title:{
        display: true,
        text: 'HTS_TST_POS',
        color: '#FAD7A0'
      },
      legend: {
          display: false,
          position: 'bottom',
          labels: {
              boxWidth: 12
          }
      }
  };
  var lineData = {
      labels: [],
      datasets: [
          {
              data: [],
              backgroundColor: '#154360',
              borderWidth: 5,
              borderColor:'#154360',
              fill:false
          }
      ]
  };
  var chart = new Chart(ctx, {
      type: 'line',
      data: lineData,
      options: lineOptions
  });
  ajax_chart(chart, hts_tst_pos_line);
});
$(function () {
  var ctx = document.getElementById("TX_NEW-LINE").getContext('2d');
  var lineOptions = {
      responsive: false,
      maintainAspectRatio: false,
      title:{
        display: true,
        text: 'TX_NEW',
        color: '#FAD7A0'
      },
      legend: {
          display: false,
          position: 'bottom',
          labels: {
              boxWidth: 12
          }
      }
  };
  var lineData = {
      labels: [],
      datasets: [
          {
              data: [],
              backgroundColor: '#154360',
              borderWidth: 5,
              borderColor:'#154360',
              fill:false
          }
      ]
  };
  var chart = new Chart(ctx, {
      type: 'line',
      data: lineData,
      options: lineOptions
  });
  ajax_chart(chart, tx_new_line);
});
$(function () {
    var ctx = document.getElementById("TX_NET_NEW-LINE").getContext("2d");
    var lineOptions = {
        responsive: false,
        maintainAspectRatio: false,
        title:{
          display: true,
          text: 'TX_NET_NEW',
          color: '#FAD7A0'
        },
        legend: {
            display: false,
            position: 'bottom',
            labels: {
                boxWidth: 12
            }
        }
    };
    var lineData = {
        labels: [],
        datasets: [
            {
                data: [],
                backgroundColor: '#154360',
                borderWidth: 5,
                borderColor:'#154360',
                fill:false
            }
        ]
    };
    var chart = new Chart(ctx, {
        type: 'line',
        data: lineData,
        options: lineOptions
    });
    ajax_chart(chart, tx_net_new_line);
});

function percentage(p1, p2){
  return [(p1/p2*100.00).toFixed(3)];
}
function ajax_overlayBar_chart(chart, url, data) {
    var data = data || {};
    $.getJSON(url, data).done(function(response) {
        chart.data.datasets[0].data = percentage(response.data.min, response.data.max);  // or you can iterate for multiple datasets
        chart.data.datasets[1].data = percentage(response.data.max, response.data.max); // or you can iterate for multiple datasets
        chart.update(); // finally update our chart
    });
}
function ajax_overlay_chart(chart, url, data) {
    var data = data || {};
    $.getJSON(url, data).done(function(response) {
        chart.data.labels = response.labels;
        chart.data.datasets[0].data = response.data.first; // or you can iterate for multiple datasets
        chart.data.datasets[1].data = response.data.second; // or you can iterate for multiple datasets
        chart.update(); // finally update our chart
    });
}
function ajax_chart(chart, url, data) {
    var data = data || {};
    $.getJSON(url, data).done(function(response) {
        chart.data.labels = response.labels;
        chart.data.datasets[0].data = response.data.quantity; // or you can iterate for multiple datasets
        chart.update(); // finally update our chart
    });
}
