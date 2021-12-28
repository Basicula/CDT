const CANVAS_ID = "draw_field";
const POINT_RADIUS = 3;
const MODS = {DOT:"dot", CONSTRAINT:"constraint", NONE:"none"};

var canvas, context;
var constraints = [];
var active_constraint = [];
var active_constraint_end;
var mode = MODS.NONE;
var points, triangles = [];

function init_canvas() {
    canvas = document.getElementById(CANVAS_ID);
    context = canvas.getContext("2d");
    canvas.height = window.innerHeight;
    canvas.width  = window.innerWidth;
    points = [];
    canvas.onmousedown = function(e) {
        var x = e.clientX;
        var y = e.clientY;
        points.push([x, y]);
        
        if (mode == MODS.NONE) {
            if (e.ctrlKey) {
                mode = MODS.CONSTRAINT;
                active_constraint = [[x, y]];
                
                this.onmousemove = function(e) {
                    if (mode == MODS.CONSTRAINT) {
                        if (!e.ctrlKey) {
                            mode = MODS.NONE;
                            constraints.push(active_constraint);
                            add_constraint(active_constraint, true);
                            this.onmousemove = function(){};
                            active_constraint = []
                            active_constraint_end = [];
                        }
                        else {
                            active_constraint_end = [e.clientX, e.clientY];
                            redraw();
                        }
                    }
                }
            }
            else
                mode = MODS.DOT;
        }
        else if (mode == MODS.CONSTRAINT) {
            active_constraint.push([x, y]);
            redraw();
        }

        this.onmouseup = function() {
            if (mode == MODS.DOT) {
                mode = MODS.NONE;
                add_point(x, y);
            }
        }
    };
}

function add_point(x, y) {
    $.ajax({
        url: "/add_point",
        type: "PUT",
        data: {x: x, y: y},
        success: function() {
            triangulate();
        }
    });
}

function add_constraint(constraint, is_closed) {
    $.ajax({
        url: "/add_constraint",
        type: "PUT",
        data: {constraint: JSON.stringify(constraint), closed : is_closed},
        success: function() {
            triangulate();
        }
    });
}

function redraw() {
    context.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < points.length; ++i)
    draw_point(points[i]);
    draw_constraints();
    for (let i = 0; i < triangles.length; ++i)
        draw_triangle(triangles[i]);
}

function triangulate() {
    $.ajax({
        url: "/triangulate",
        type: "POST",
        success: function(response) {
            triangles = response.triangles;
            redraw();
        }
    });
}

function draw_constraint(constraint, is_closed) {
    context.beginPath();
    context.moveTo(constraint[0][0], constraint[0][1]);
    for (let point_id = 1; point_id < constraint.length; ++point_id)
        context.lineTo(constraint[point_id][0], constraint[point_id][1]);
    if (is_closed)
        context.lineTo(constraint[0][0], constraint[0][1]);
    context.strokeStyle = "black";
    context.lineWidth = 3;
    context.stroke();
}

function draw_constraints() {
    for (let i = 0; i < constraints.length; ++i) {
        const constraint = constraints[i];
        draw_constraint(constraint, true);
    }
    
    if (active_constraint.length > 0) {
        draw_constraint(active_constraint, false);
        context.beginPath();
        const last_point_id = active_constraint.length - 1;
        context.moveTo(active_constraint[last_point_id][0], active_constraint[last_point_id][1]);
        context.lineTo(active_constraint_end[0], active_constraint_end[1]);
        context.stroke();
    }
}

function draw_point(point) {
    context.beginPath();
    context.arc(point[0], point[1], POINT_RADIUS, 0, Math.PI * 2, true);
    context.fillStyle = "red";
    context.fill();
}

function draw_triangle(triangle) {
    context.beginPath();
    context.moveTo(points[triangle[0]][0], points[triangle[0]][1]);
    context.lineTo(points[triangle[1]][0], points[triangle[1]][1]);
    context.lineTo(points[triangle[2]][0], points[triangle[2]][1]);
    context.lineTo(points[triangle[0]][0], points[triangle[0]][1]);
    context.strokeStyle = "black";
    context.lineWidth = 1;
    context.stroke()
}

document.addEventListener("DOMContentLoaded", init_canvas);
window.onload = function() {
    $.ajax({
        url: "/clear",
        type: "POST"
    });
}