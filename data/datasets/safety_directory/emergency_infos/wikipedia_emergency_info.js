/**
 * Developer console script used to generate the associated json file.
 * Wikipedia URL : https://en.wikipedia.org/wiki/List_of_suicide_crisis_lines
 * Author : Lucas Oulieu
 */

let datas = {};
const all = Array.from(document.querySelectorAll("table.wikitable tr"));

//Removing the first element because it is the Header of the Table
all.shift();

all.forEach((tr) => {
  country = tr.querySelectorAll("th a")[1].title;
  datas[country] = {
    img: tr.querySelector("th a").href,
    lines: Array.from(tr.querySelectorAll("td li")).map((tr) => tr.textContent),
  };
});
const jsonDatas = JSON.stringify(datas);
console.log(jsonDatas);

const element = document.createElement("a");
element.setAttribute(
  "href",
  "data:text/plain;charset=utf-8," + encodeURIComponent(jsonDatas)
);
element.setAttribute("download", "wikipedia_emergency_info.json");

element.style.display = "none";
document.body.appendChild(element);

element.click();

document.body.removeChild(element);
